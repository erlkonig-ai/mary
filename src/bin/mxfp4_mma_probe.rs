//! `mxfp4_mma_probe` — ask *this* GPU which block-scaled FP4 tensor-core
//! shapes it actually reports, instead of reading a capability table.
//!
//! [`mary::nn::mxfp4`]'s transcode exists to land 4-bit weights on a
//! block-scaled MMA. Which encoding to land them *in* is a property of the
//! device, not of the format: sm_120-class hardware exposes `mma.sync
//! ...kind::mxf4nvf4.block_scale`, and the scale operand can be either a
//! `ue8m0` at `scale_vec::2X` (one scale per 32 elements — MXFP4, the
//! checkpoint's own layout) or an `e4m3` at `scale_vec::4X` (one per 16 —
//! NVFP4, the transcode's target). Assuming which one is present is exactly
//! the sort of premise that costs a day, so this prints what the runtime
//! enumerates on the machine it runs on.
//!
//! ## Gate
//!
//! An empty feature set is ambiguous — it could mean "this GPU has no scaled
//! MMA" or "the query never reached a device". So before reporting anything,
//! the probe requires the *plain* (unscaled) MMA set to be non-empty. That set
//! is populated for every CUDA arch from sm_70 up, so its presence proves the
//! runtime enumerated a real device and any absence below is a real absence.
//!
//! Build: `--features cuda-backend,mxfp4`.

use cubecl::ir::{ElemType, FloatKind, StorageType};
use cubecl::Runtime;

use mary::nn::mxfp4::{MX_BLOCK, NV_BLOCK};

type Rt = cubecl::cuda::CudaRuntime;

/// Is this storage type packed E2M1 (`e2m1x2`), the 4-bit weight element both
/// MXFP4 and NVFP4 use?
fn is_fp4(t: &StorageType) -> bool {
    matches!(
        t,
        StorageType::Packed(ElemType::Float(FloatKind::E2M1), _)
            | StorageType::Scalar(ElemType::Float(FloatKind::E2M1))
    )
}

fn main() {
    let device = Default::default();
    let client = Rt::client(&device);
    let props = client.properties();
    let features = &props.features;

    // ---- gate: prove the query reached a device -------------------------
    if features.matmul.mma.is_empty() {
        eprintln!(
            "GATE FAILED — the runtime reports no plain MMA combinations at all, so an empty \
             scaled-MMA set would say nothing about the hardware. No capability reported."
        );
        std::process::exit(1);
    }
    println!(
        "gate ok — runtime enumerated a device: {} plain MMA combinations reported\n",
        features.matmul.mma.len()
    );

    println!("block-scaled MMA combinations with 4-bit (E2M1) operands:");
    println!("{:<10} {:>4} {:>4} {:>4} {:>8} {:>16}", "scales", "m", "n", "k", "scale_vec", "elems/scale");
    let mut mx = false;
    let mut nv = false;
    for c in &features.matmul.scaled_mma {
        if !is_fp4(&c.a_type) || !is_fp4(&c.b_type) {
            continue;
        }
        let per_scale = c.k / c.scales_factor;
        let scales = format!("{:?}", c.scales_type);
        println!(
            "{:<10} {:>4} {:>4} {:>4} {:>8} {:>16}",
            scales.replace("Scalar(Float(", "").replace("))", ""),
            c.m,
            c.n,
            c.k,
            format!("{}X", c.scales_factor),
            per_scale
        );
        if per_scale as usize == MX_BLOCK {
            mx = true;
        }
        if per_scale as usize == NV_BLOCK {
            nv = true;
        }
    }

    println!();
    println!("MXFP4 shape ({MX_BLOCK} elements/scale) available: {mx}");
    println!("NVFP4 shape ({NV_BLOCK} elements/scale) available: {nv}");
    println!(
        "{} scaled-MMA combinations total, {} of them 4-bit.",
        features.matmul.scaled_mma.len(),
        features.matmul.scaled_mma.iter().filter(|c| is_fp4(&c.a_type) && is_fp4(&c.b_type)).count()
    );
}
