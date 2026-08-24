//! The one place a Burn tensor and a raw cubecl handle are admitted to be the
//! same bytes.
//!
//! Inkling's forward is written in two dialects and has to be. The routed
//! experts go through `mma.sync…kind::mxf4nvf4` and `mma.sync…bf16`, which are
//! hand-written cubecl kernels taking `cubecl::server::Handle`; everything
//! around them — attention, the shared and dense MLPs, the norms, the short
//! convolutions — is Burn. As long as the two dialects could only meet on the
//! host, the residual stream had to come DOWN between them, and it did: twice
//! a layer, forty-two layers, every token.
//!
//! They do not actually differ. `burn::backend::Cuda<f32>` is
//! `CubeBackend<CudaRuntime, …>`, its float primitive IS a `CubeTensor`, and a
//! `CubeTensor`'s `handle` field is exactly the `Handle` the kernels take. So
//! this module is two functions and no work: take the handle out, put a handle
//! back in. The same seam `qwen3tts::megakernel` opens on wgpu, for the same
//! reason.
//!
//! What it is NOT is a general conversion. Both functions insist on contiguous
//! f32, because a kernel that indexes `[row * k + col]` and a tensor that has
//! been transposed into non-unit strides disagree silently and produce plausible
//! numbers. [`handle_of`] makes the tensor contiguous rather than asserting,
//! since Burn hands back stride bookkeeping on size-1 dims that is contiguous
//! in every sense but the predicate's.

use burn::tensor::{DType, Tensor, TensorPrimitive};
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::ComputeClient;
use cubecl::server::Handle;

/// The backend this seam is written for. Not generic: the whole point is that
/// these two types are the same on THIS runtime, and a generic version would
/// have to be able to fail.
pub type Bk = burn::backend::Cuda<f32>;

/// The device buffer behind a Burn f32 tensor, as the handle the raw kernels
/// take.
///
/// Consumes the tensor. It does not have to — `Handle` is refcounted and the
/// buffer outlives either wrapper — but taking it by value says what is true:
/// after this call there are two names for one allocation, and writing through
/// one of them is visible through the other.
pub fn handle_of<const D: usize>(t: Tensor<Bk, D>) -> Handle {
    match t.into_primitive() {
        TensorPrimitive::Float(c) => {
            let mut c: CubeTensor<CudaRuntime> = c;
            if !c.is_contiguous() {
                c = burn_cubecl::kernel::into_contiguous(c);
            }
            assert_eq!(c.dtype, DType::F32, "the inkling seam is f32 on both sides");
            c.handle
        }
        TensorPrimitive::QFloat(_) => {
            panic!("a quantized Burn tensor has no plain f32 buffer to hand over")
        }
    }
}

/// The device buffer behind a Burn float tensor, and the dtype it is in.
///
/// [`handle_of`] asserts f32 because the kernels on its far side index bytes at
/// a fixed width, and for as long as every one of them did, "the inkling seam
/// is f32 on both sides" was a true sentence and a useful guard. It is not true
/// any more: the residual stream is BF16 on the narrow lane, and the kernels
/// that read it are generic over their element type precisely so that they can.
///
/// So this is the same seam with the assertion turned into a RETURN VALUE. A
/// caller that gets the dtype back has to decide what to do with it — which is
/// the point, because the decision is exactly the one the assertion used to
/// make on everybody's behalf, and the two lanes want different answers.
/// [`handle_of`] stays where it is for every caller that genuinely only has an
/// f32 kernel.
pub fn handle_of_any<const D: usize>(t: Tensor<Bk, D>) -> (Handle, DType) {
    match t.into_primitive() {
        TensorPrimitive::Float(c) => {
            let mut c: CubeTensor<CudaRuntime> = c;
            if !c.is_contiguous() {
                c = burn_cubecl::kernel::into_contiguous(c);
            }
            let dt = c.dtype;
            (c.handle, dt)
        }
        TensorPrimitive::QFloat(_) => {
            panic!("a quantized Burn tensor has no plain float buffer to hand over")
        }
    }
}

/// The dtype of a Burn float tensor, without consuming it.
///
/// The narrow lane branches on this in a dozen places and every one of them
/// still needs the tensor afterwards, so the borrowing form is the one that
/// gets used.
pub fn dtype_of<const D: usize>(t: &Tensor<Bk, D>) -> DType {
    match &t.clone().into_primitive() {
        TensorPrimitive::Float(c) => c.dtype,
        TensorPrimitive::QFloat(_) => panic!("a quantized Burn tensor has no plain float dtype"),
    }
}

/// The same, for an `Int` tensor.
///
/// A separate function rather than a generic one because Burn spells the two
/// differently: a float tensor's primitive is the `TensorPrimitive` enum (it
/// may be quantized), an integer tensor's is the `CubeTensor` outright. The
/// dtype assertion is the one that matters — the routed lane's row indices are
/// `i32` and a kernel reading them as `i64` would index somewhere else
/// entirely.
pub fn int_handle_of<const D: usize>(t: Tensor<Bk, D, burn::tensor::Int>) -> Handle {
    let mut c: CubeTensor<CudaRuntime> = t.into_primitive();
    if !c.is_contiguous() {
        c = burn_cubecl::kernel::into_contiguous(c);
    }
    assert_eq!(c.dtype, DType::I32, "the inkling seam indexes with i32");
    c.handle
}

/// The inverse: a `[rows, cols]` f32 device buffer, as a Burn tensor.
///
/// `client` and `device` are asked for rather than derived, because a `Handle`
/// does not know which runtime allocated it and guessing would be a way to
/// build a tensor pointing into another device's memory.
pub fn tensor_of(
    client: ComputeClient<CudaRuntime>,
    device: burn::backend::cuda::CudaDevice,
    handle: Handle,
    rows: usize,
    cols: usize,
) -> Tensor<Bk, 2> {
    tensor_of_dt(client, device, handle, rows, cols, DType::F32)
}

/// [`tensor_of`] at a named dtype.
///
/// The dtype is the caller's because it is the KERNEL's: a buffer is whatever
/// the kernel that wrote it stored, and nothing about the handle records that.
/// Getting it wrong is not a type error, it is a tensor that reads two BF16
/// values as one f32 and keeps going.
pub fn tensor_of_dt(
    client: ComputeClient<CudaRuntime>,
    device: burn::backend::cuda::CudaDevice,
    handle: Handle,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Tensor<Bk, 2> {
    let c = CubeTensor::<CudaRuntime>::new_contiguous(
        client,
        device,
        [rows, cols].into(),
        handle,
        dtype,
    );
    Tensor::from_primitive(TensorPrimitive::Float(c))
}

/// A rank-3 f32 tensor as `(handle, strides)`, WITHOUT making it contiguous.
///
/// [`handle_of`] copies when a tensor is not contiguous, and it is right to: a
/// kernel that indexes `[row * k + col]` cannot read a strided buffer. But
/// Burn's f32 matmul does not return a contiguous tensor -- it returns a PADDED
/// one, `[32, 7000, 7000]` with strides `[49_280_000, 7040, 1]`, the row
/// rounded up to a multiple of 64 -- and copying 6.3 GiB per layer to remove
/// forty columns of padding cost 504 ms of a 5.47 s pass at 7000 tokens, 9.2%
/// of it, for nothing. A kernel that is told the row stride does not need the
/// copy, which is why this exists beside `handle_of` rather than inside it:
/// only a caller knows whether its kernel can honour a stride.
///
/// The innermost stride must be 1. Anything else is a permutation rather than
/// padding, and this is not the function for it.
pub fn strided_of3(t: Tensor<Bk, 3>) -> (Handle, [usize; 3]) {
    match t.into_primitive() {
        TensorPrimitive::Float(c) => {
            let c: CubeTensor<CudaRuntime> = c;
            assert_eq!(c.dtype, DType::F32, "the inkling seam is f32 on both sides");
            let st = c.meta.strides.clone();
            assert_eq!(st.rank(), 3, "strided_of3 wants rank 3");
            assert_eq!(st[2], 1, "the innermost stride is {}, not 1", st[2]);
            let strides = [st[0], st[1], st[2]];
            (c.handle, strides)
        }
        TensorPrimitive::QFloat(_) => panic!("a quantized Burn tensor has no plain f32 buffer"),
    }
}

/// The inverse of [`strided_of3`]: the same buffer and the same strides, as a
/// Burn tensor again.
pub fn tensor_strided3(
    client: ComputeClient<CudaRuntime>,
    device: burn::backend::cuda::CudaDevice,
    handle: Handle,
    shape: [usize; 3],
    strides: [usize; 3],
) -> Tensor<Bk, 3> {
    // Built contiguous and then told the truth about its strides. Naming
    // `Metadata` here would mean depending on `burn-std` for one type; the
    // field is public and this is the same two words.
    let mut c =
        CubeTensor::<CudaRuntime>::new_contiguous(client, device, shape.into(), handle, DType::F32);
    c.meta.strides = strides.into();
    Tensor::from_primitive(TensorPrimitive::Float(c))
}

/// The same, for a rank-3 buffer.
///
/// Not `tensor_of(...).reshape([d0, d1, d2])`: Burn's `reshape` decides between
/// rewriting the strides and COPYING, and on a `[heads * n, n]` score matrix it
/// chose the copy -- 6.3 GiB read and written per layer per pass at 7000
/// tokens, 480 ms of a 5.5 s pass, for a shape change that moves no bytes.
/// Building the tensor at the rank it is wanted at asks the question once.
pub fn tensor_of3(
    client: ComputeClient<CudaRuntime>,
    device: burn::backend::cuda::CudaDevice,
    handle: Handle,
    d0: usize,
    d1: usize,
    d2: usize,
) -> Tensor<Bk, 3> {
    let c = CubeTensor::<CudaRuntime>::new_contiguous(
        client,
        device,
        [d0, d1, d2].into(),
        handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(c))
}

/// The compute client a Burn tensor was allocated on.
///
/// The forward needs one to launch the raw kernels with, and taking it from a
/// tensor that is already on the device is the only way to be sure it is the
/// SAME client — two `CudaRuntime::client(&Default::default())` calls are
/// meant to return the same one, and "meant to" is not a thing to bet a
/// pointer on.
pub fn client_of<const D: usize>(t: &Tensor<Bk, D>) -> ComputeClient<CudaRuntime> {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(c) => c.client,
        TensorPrimitive::QFloat(_) => panic!("quantized"),
    }
}

/// What the device pool has RESERVED, in bytes.
///
/// The one number the admission gate is trying to predict. On a unified-memory
/// part `cuMemAlloc` is node memory, so the pool's reservation IS the run's
/// activation footprint -- and unlike `MemAvailable` it belongs to this process
/// alone, so a box with something else running on it does not move it.
///
/// Zero when the runtime will not answer, which reads as "nothing to compare
/// against" wherever it is printed rather than as a suspiciously small run.
pub fn pool_reserved(client: &ComputeClient<CudaRuntime>) -> u64 {
    client.memory_usage().map(|u| u.bytes_reserved).unwrap_or(0)
}

/// What the device pool has RESERVED against what the run is holding.
///
/// The gap between those two is not padding. cubecl's sliced pools hand back a
/// page only when it is ENTIRELY free (`SlicedPool::cleanup`), so one surviving
/// slice keeps its whole page, and a long-lived tensor born in the middle of a
/// burst of transient ones strands everything that shared its page.
/// `memory_cleanup` cannot help with that — it is the call that just tried.
///
/// Worth printing because the growth is invisible in every other number the run
/// reports. `cuMemAlloc` is driver memory on this part, so a pool that has grown
/// tens of GiB shows up as a machine with no memory left and a process whose
/// resident size did not move at all.
pub fn pool_line(client: &ComputeClient<CudaRuntime>, at: &str) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    match client.memory_usage() {
        Ok(u) => {
            format!(
            "    pool[{at}]: {:.2} GiB reserved, {:.2} live, {:.2} padding, {:.2} GiB STRANDED \
             over {} slices",
            u.bytes_reserved as f64 / GIB,
            u.bytes_in_use as f64 / GIB,
            u.bytes_padding as f64 / GIB,
            u.bytes_reserved.saturating_sub(u.bytes_in_use + u.bytes_padding) as f64 / GIB,
            u.number_allocs,
        )
        }
        Err(e) => format!("    pool[{at}]: unavailable ({e:?})"),
    }
}

/// `t` with a unit-stride layout, at whatever float dtype it already carries.
///
/// [`handle_of`] does this on the way out to a raw kernel, and it is the only
/// way the Burn lane had to ask for it — but it also asserts f32, because the
/// kernels on the far side index bytes. A tensor that stays in Burn has no such
/// obligation: it can be BF16, and the reason to make it contiguous is the
/// reason the attention lane already had, which is that `matmul` makes its
/// operands contiguous itself and doing it once per LAYER is cheaper than once
/// per query block.
///
/// So this is `handle_of`'s copy without `handle_of`'s dtype: same
/// `into_contiguous`, same no-op when the layout is already unit-stride, and
/// the tensor comes back as a tensor rather than as a pointer.
pub fn contiguous<const D: usize>(t: Tensor<Bk, D>) -> Tensor<Bk, D> {
    match t.into_primitive() {
        TensorPrimitive::Float(c) => {
            let mut c: CubeTensor<CudaRuntime> = c;
            if !c.is_contiguous() {
                c = burn_cubecl::kernel::into_contiguous(c);
            }
            Tensor::from_primitive(TensorPrimitive::Float(c))
        }
        TensorPrimitive::QFloat(_) => {
            panic!("a quantized Burn tensor has no plain float buffer to lay out")
        }
    }
}
