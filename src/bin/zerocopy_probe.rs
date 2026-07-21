//! One-tensor proof of the zero-copy mmap->GPU aliasing seam (Phase 2).
//!
//! Allocates a page-aligned host buffer with known f32 values, registers it as a
//! GPU tensor via the patched `ComputeClient::register_external_aliased` (Metal
//! `newBufferWithBytesNoCopy:` over the host pages — NO copy, NO allocation),
//! reads it back, and asserts the bytes match. Then MUTATES the host buffer and
//! re-reads to prove it's a true alias (the GPU sees host writes), not a copy.
//!
//!   cargo run --release --features gemma --bin zerocopy_probe
//!
//! macOS / Metal only (unified memory).

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use std::alloc::{alloc, dealloc, Layout};

const PAGE: usize = 16384; // Apple Silicon page size
const N: usize = 256; // f32 values
const NBYTES: usize = N * 4;

fn main() {
    assert!(NBYTES <= PAGE, "test tensor must fit one page");
    let layout = Layout::from_size_align(PAGE, PAGE).unwrap();
    // SAFETY: non-zero layout; freed at the end after all GPU use.
    let ptr = unsafe { alloc(layout) } as *mut f32;
    assert!(!ptr.is_null());

    // Fill with a known ramp.
    let host: Vec<f32> = (0..N).map(|i| i as f32 * 1.5 - 7.0).collect();
    // SAFETY: ptr owns PAGE bytes >= NBYTES.
    unsafe { std::ptr::copy_nonoverlapping(host.as_ptr(), ptr, N) };

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    // Register the host page as an aliased GPU buffer (offset 0, size NBYTES).
    // SAFETY: ptr is page-aligned, PAGE bytes, lives until end of main.
    let handle = unsafe {
        client.register_external_aliased(
            ptr as *mut core::ffi::c_void, PAGE as u64, 0, NBYTES as u64,
            std::sync::Arc::new(()),
        )
    };

    // Read it back off the GPU.
    let bytes = client.read_one(handle.clone()).expect("read aliased handle");
    let got = as_f32(&bytes);
    let ok1 = got.len() >= N && got[..N] == host[..];
    println!("[alias] read-back matches host ramp: {ok1}  (got[0..3]={:?}, host[0..3]={:?})", &got[..3], &host[..3]);

    // Prove it's a TRUE alias: mutate host memory, re-read, expect the change.
    // SAFETY: same region, still alive, no concurrent GPU op in flight.
    unsafe { *ptr.add(0) = 999.0; *ptr.add(N - 1) = -999.0; }
    let bytes2 = client.read_one(handle).expect("re-read aliased handle");
    let got2 = as_f32(&bytes2);
    let ok2 = (got2[0] - 999.0).abs() < 1e-3 && (got2[N - 1] + 999.0).abs() < 1e-3;
    println!("[alias] host mutation visible on GPU (true zero-copy alias): {ok2}  (got2[0]={}, got2[last]={})", got2[0], got2[N - 1]);

    // SAFETY: all GPU reads done; release the host allocation.
    unsafe { dealloc(ptr as *mut u8, layout) };

    if ok1 && ok2 {
        println!("=== PASS — zero-copy mmap->GPU aliasing works end-to-end ===");
    } else {
        println!("=== FAIL ===");
        std::process::exit(1);
    }
}

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
