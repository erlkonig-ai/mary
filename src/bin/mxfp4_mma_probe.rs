//! `mxfp4_mma_probe` — report which block-scaled FP4 tensor-core shapes
//! **CubeCL believes** this architecture supports.
//!
//! ## Read this before quoting its output
//!
//! An earlier version of this file claimed to "ask *this* GPU ... instead of
//! reading a capability table". That was wrong, and the wrongness was load
//! bearing: it was used to argue that the checkpoint's own MXFP4 might reach
//! the tensor cores with no transcode at all.
//!
//! What actually happens: creating the client does touch the device, but only
//! to read its architecture version. The feature sets below are then produced
//! by `supported_scaled_mma_combinations(arch)`, a **pure function of that
//! version number** — no `cuDeviceGetAttribute`, no `ptxas` probe, no kernel.
//! The rows printed for sm_121 follow deterministically from `120 <= 121 < 130`
//! and would print identically on any sm_12x, working or not. `ptxas` is not
//! even installed on this machine.
//!
//! The gate below inherits the same flaw. It requires the *plain* MMA set to be
//! non-empty as evidence that "the query reached a device" — but that set comes
//! from the same version-keyed table, so its non-emptiness proves only that an
//! architecture version was parsed. It cannot distinguish a real capability
//! from a table entry.
//!
//! So this binary answers "what does CubeCL think sm_121 can do", which is
//! useful for choosing a code path and worthless as evidence about silicon.
//!
//! ## What would actually settle it
//!
//! Install `ptxas`, assemble a block-scaled FP4 `mma.sync` with a `ue8m0` scale
//! operand at `scale_vec::2X`, launch it on the GB10, and compare against a CPU
//! reference. If MXFP4 executes natively, [`mary::nn::mxfp4`]'s NVFP4 transcode
//! is unnecessary; if it does not, the transcode is on the critical path. That
//! measurement has NOT been made.
//!
//! (For contrast: NVFP4 matmul on sm_121 *was* verified by launching a real
//! kernel against a CPU reference. That result stands. This one is a table.)
//!
//! Build: `--features cuda-backend,mxfp4`.

use cubecl::Runtime;
use cubecl::ir::{ElemType, FloatKind, StorageType};

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

    // ---- gate: NOT a proof of hardware capability ------------------------
    // This establishes only that an architecture version was parsed. Both sets
    // below are pure functions of that version, so neither says anything about
    // what the silicon executes. Kept because an empty plain set would mean the
    // runtime failed entirely, which is still worth catching.
    if features.matmul.mma.is_empty() {
        eprintln!(
            "GATE FAILED — the runtime reports no plain MMA combinations at all, so an empty \
             scaled-MMA set would say nothing about the hardware. No capability reported."
        );
        std::process::exit(1);
    }
    println!(
        "gate ok — an architecture version was parsed: {} plain MMA combinations listed.\n  NOTE: these are CubeCL's version-keyed table entries, NOT a device capability\n  query. Nothing below is evidence about silicon; see the module doc.\n",
        features.matmul.mma.len()
    );

    println!("block-scaled MMA combinations with 4-bit (E2M1) operands:");
    println!(
        "{:<10} {:>4} {:>4} {:>4} {:>8} {:>16}",
        "scales", "m", "n", "k", "scale_vec", "elems/scale"
    );
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
        features
            .matmul
            .scaled_mma
            .iter()
            .filter(|c| is_fp4(&c.a_type) && is_fp4(&c.b_type))
            .count()
    );
}
