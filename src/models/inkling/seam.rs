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
