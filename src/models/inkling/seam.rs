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
    let c = CubeTensor::<CudaRuntime>::new_contiguous(
        client,
        device,
        [rows, cols].into(),
        handle,
        DType::F32,
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
    let mut c = CubeTensor::<CudaRuntime>::new_contiguous(
        client,
        device,
        shape.into(),
        handle,
        DType::F32,
    );
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
