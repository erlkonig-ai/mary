//! Custom Metal device setup that raises `max_storage_buffer_binding_size` past
//! wgpu's default 4 GiB clamp.
//!
//! wgpu-core's `request_device` path validates `required_limits` against
//! adapter caps via `check_limits`, and Metal HAL's own `open()` still clamps
//! `max_storage_buffer_binding_size` to `u32::MAX`. The `create_device_from_hal`
//! path does NOT run `check_limits`, and Metal HAL's `open()` actually ignores
//! the `_limits` parameter entirely — so whatever we put in the
//! `DeviceDescriptor` is what `device.limits()` reports back. That's what
//! cubecl's memory layer reads to decide buffer sizing.
//!
//! Upstream hint from Genna (cubecl maintainer) — same approach as cubecl's
//! Vulkan `from_native` path (cubecl-wgpu/src/backend/vulkan.rs:39-168).
//! Works since wgpu 29 bumped `max_storage_buffer_binding_size` to `u64`.

use burn::backend::wgpu::WgpuDevice;
#[cfg(target_os = "macos")]
use burn::backend::wgpu::{init_device, RuntimeOptions, WgpuSetup};
#[cfg(target_os = "macos")]
use pollster::block_on;
#[cfg(target_os = "macos")]
use wgpu::hal::{self, Adapter as HalAdapter};

/// Build a `WgpuDevice` backed by Metal with `max_storage_buffer_binding_size`
/// raised to `buffer_binding_size`. Panics if no Metal adapter is available.
#[cfg(target_os = "macos")]
pub fn init_metal_device_with_large_buffers(buffer_binding_size: u64) -> WgpuDevice {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no Metal adapter available");

    let features = adapter
        .features()
        .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);

    let mut limits = adapter.limits();
    limits.max_storage_buffer_binding_size =
        buffer_binding_size.max(limits.max_storage_buffer_binding_size);
    limits.max_buffer_size = buffer_binding_size.max(limits.max_buffer_size);

    let memory_hints = wgpu::MemoryHints::MemoryUsage;

    let hal_device = unsafe {
        let hal_adapter = adapter
            .as_hal::<hal::api::Metal>()
            .expect("adapter is not Metal");
        hal_adapter
            .open(features, &limits, &memory_hints)
            .expect("failed to open Metal HAL device")
    };

    let descriptor = wgpu::DeviceDescriptor {
        label: Some("gaze-metal-raised-limits"),
        required_features: features,
        required_limits: limits,
        memory_hints,
        trace: wgpu::Trace::Off,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    };

    let (device, queue) = unsafe {
        adapter
            .create_device_from_hal::<hal::api::Metal>(hal_device, &descriptor)
            .expect("create_device_from_hal failed")
    };

    let setup = WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: wgpu::Backend::Metal,
    };

    init_device(setup, RuntimeOptions::default())
}

/// Non-Apple fallback: no Metal HAL here, so no binding-size raise — returns
/// the standard default wgpu device (Vulkan/GL discovery). Models needing
/// storage bindings past wgpu's 4 GiB clamp are Apple-only until an
/// equivalent Vulkan `from_native` raise lands.
#[cfg(not(target_os = "macos"))]
pub fn init_metal_device_with_large_buffers(_buffer_binding_size: u64) -> WgpuDevice {
    WgpuDevice::default()
}

/// Convenience: 16 GiB storage-buffer-binding, enough for Gemma 4 31B's
/// 5.6 GB embedding plus headroom.
pub fn init_metal_device_16gb() -> WgpuDevice {
    init_metal_device_with_large_buffers(16 * 1024 * 1024 * 1024)
}
