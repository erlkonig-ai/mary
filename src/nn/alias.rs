//! Zero-copy pile→GPU aliasing for the RAW (non-fusion) Metal backends:
//! mmap'd pile blob → Metal `newBufferWithBytesNoCopy` → cubecl handle →
//! `CubeTensor` — no host materialization, no upload, the tensor's GPU buffer
//! IS the pile's mapped pages. This is the gemma seam
//! (`crate::persist::alias_f16_leaf`) factored out and made generic over the
//! element type, so the qwen3tts raw talker lane (f16) and any future raw
//! lane share one implementation.
//!
//! Preconditions are checked and reported as `Err(reason)` — the caller
//! decides whether to warn-and-fall-back. Under the V3 pile format every
//! record's payload is 256-aligned, which satisfies Metal's buffer-binding
//! alignment; the blob's containing pages are mapped as the buffer and the
//! tensor binds at its in-page offset. The returned tensor's buffer holds the
//! pile mmap alive (`register_external_aliased` keepalive), so the pile
//! reader may be dropped after loading.
//!
//! The FUSED backends (`BFused`/`BFusedHalf`) must NOT take this route:
//! burn 0.21's fusion codegen miscompiles graphs over many externally-
//! registered buffers (see PORT_NOTES.md, "Zero-copy alias probe" — the
//! fusion-import variant of this module was probed and deleted at bf92171).
//! The raw backends have no fusion graph to miscompile; the gemma lanes have
//! run this exact seam in production since the cubecl fork landed.

use burn::backend::wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::cubecl::Runtime;
use burn_cubecl::CubeBackend;
use memmap2::MmapRaw;
use std::sync::Arc;

/// The raw cube backend family `B`/`BHalf` are aliases of.
pub type RawCube<F> = CubeBackend<WgpuRuntime, F, i32, u8>;

/// Apple-silicon page size — `newBufferWithBytesNoCopy` requires page-aligned
/// base + length, so the handle covers the blob's containing pages.
const PAGE: u64 = 16384;

/// Alias one mmap'd pile blob onto the GPU as a flat `[n]` tensor of the raw
/// backend — no copy, no host materialization. `F` must match the blob's
/// element width exactly (f16 leaves → `half::f16`, f32 leaves → `f32`).
pub fn alias_flat_raw<F>(
    bytes: anybytes::Bytes,
    device: &WgpuDevice,
) -> Result<Tensor<RawCube<F>, 1>, String>
where
    F: burn_cubecl::FloatElement,
{
    let elem = core::mem::size_of::<F>() as u64;
    let nbytes = bytes.len() as u64;
    if nbytes == 0 || !nbytes.is_multiple_of(elem) {
        return Err(format!("blob length {nbytes} is not a multiple of the {elem}-byte element"));
    }
    // wgpu storage-binding sizes must be 4-byte multiples: an odd-element f16
    // leaf (nbytes ≡ 2 mod 4) must fall back to upload, not panic in the driver.
    if !nbytes.is_multiple_of(4) {
        return Err(format!("blob length {nbytes} is not a 4-byte multiple (wgpu binding size)"));
    }
    let blob_ptr = bytes.as_ptr() as u64;
    if !blob_ptr.is_multiple_of(256) {
        return Err(format!("blob not 256-aligned (ptr % 256 = {})", blob_ptr % 256));
    }
    // The owner downcast = capability check (mmap-backed?) + region bounds +
    // keepalive, exactly as in the gemma seam.
    let mmap = bytes
        .downcast_to_owner::<MmapRaw>()
        .map_err(|_| "blob is not mmap-backed (in-memory store?)".to_string())?;
    let region_end = mmap.as_ptr() as u64 + mmap.len() as u64;
    let page_start = blob_ptr & !(PAGE - 1);
    let off_in_page = blob_ptr - page_start;
    let page_len = ((blob_ptr + nbytes + PAGE - 1) & !(PAGE - 1)).min(region_end) - page_start;
    let keepalive: Arc<dyn std::any::Any + Send + Sync> = mmap.clone();

    let n = (nbytes / elem) as usize;
    let dtype = <F as burn::tensor::Element>::dtype();
    let client = WgpuRuntime::client(device);
    // SAFETY: page_start/page_len is a page-aligned superset of the blob,
    // inside the (page-aligned) mmap which `keepalive` pins for the buffer's
    // life. Pile blobs are immutable (append-only file, content-addressed).
    let handle = unsafe {
        client.register_external_aliased(
            page_start as *mut core::ffi::c_void,
            page_len,
            off_in_page,
            nbytes,
            keepalive,
        )
    };
    let cube = CubeTensor::<WgpuRuntime>::new_contiguous(
        client.clone(),
        device.clone(),
        [n].into(),
        handle,
        dtype,
    );
    Ok(Tensor::from_primitive(TensorPrimitive::Float(cube)))
}
