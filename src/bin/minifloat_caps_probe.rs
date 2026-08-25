//! `minifloat_caps_probe` — what does *this* GPU tell CubeCL it can do with the
//! narrow float types NVFP4/MXFP4 are built out of?
//!
//! CubeCL gates every minifloat conversion behind a runtime capability set:
//! `T::supported_uses(&client)` returns an `EnumSet<TypeUsage>` (`Conversion`,
//! `Arithmetic`, `DotProduct`, `Buffer`) that the CUDA runtime fills in from the
//! device's compute capability. `cubecl-core`'s own `runtime_tests::minifloat`
//! skips itself when `Conversion` is missing, so the same query is the honest
//! way to ask whether a hardware encode path exists at all before writing a
//! kernel that assumes one.
//!
//! This prints the set for every type the NVFP4 and MXFP4 recipes touch —
//! `e2m1` and its packed pair `e2m1x2` (the codes), `e4m3` (the NVFP4 block
//! scale), `ue8m0` (the MXFP4 block scale), plus `e5m2`, `e2m3` and `e3m2` for
//! context — and *separately* runs a one-element kernel that actually performs
//! `f32 -> e2m1x2 -> f16x2`, because a reported capability is a claim about the
//! device and executing the cast is the evidence. A capability that is reported
//! but does not survive `Cast::cast_from` is the interesting failure, and this
//! binary is arranged so the two cannot be confused for each other.
//!
//! Read-only, one 8-byte buffer, no sweeps: safe on a contended box.

use cubecl::ir::TypeUsage;
use cubecl::prelude::*;
use cubecl::{e2m1, e2m1x2, e2m3, e3m2, e4m3, e5m2, ue8m0};

type Rt = cubecl::cuda::CudaRuntime;

/// Round-trip one f32 pair through the packed FP4 pair type, so the reported
/// `Conversion` capability is backed by an executed cast rather than a flag.
#[cube(launch_unchecked)]
fn e2m1_roundtrip(input: &Array<f32>, out: &mut Array<f32>) {
    if ABSOLUTE_POS == 0 {
        // One element at a time: `e2m1` is the scalar view of the same storage
        // and is what a quantizer kernel would emit per element.
        out[0] = f32::cast_from(e2m1::cast_from(input[0]));
        out[1] = f32::cast_from(e2m1::cast_from(input[1]));
    }
}

fn uses<T: CubePrimitive>(name: &str, client: &cubecl::client::ComputeClient<Rt>) {
    let set = T::supported_uses(client);
    let mut named: Vec<&str> = Vec::new();
    for use_ in set.iter() {
        named.push(match use_ {
            TypeUsage::Conversion => "Conversion",
            TypeUsage::Arithmetic => "Arithmetic",
            TypeUsage::DotProduct => "DotProduct",
            TypeUsage::Buffer => "Buffer",
        });
    }
    if named.is_empty() {
        named.push("(none)");
    }
    println!("  {name:<10} {}", named.join(" | "));
}

fn main() {
    let device = Default::default();
    let client = Rt::client(&device);

    println!("device properties");
    let props = client.properties();
    println!("  plane size     {:?}", props.hardware.plane_size_min);
    println!("  max shared mem {}", props.hardware.max_shared_memory_size);

    println!("\nTypeUsage reported by this device");
    uses::<e2m1>("e2m1", &client);
    uses::<e2m1x2>("e2m1x2", &client);
    uses::<e4m3>("e4m3", &client);
    uses::<e5m2>("e5m2", &client);
    uses::<ue8m0>("ue8m0", &client);
    uses::<e2m3>("e2m3", &client);
    uses::<e3m2>("e3m2", &client);

    // The claim, executed. 1.3 is not representable in E2M1; the nearest codes
    // are 1.0 and 1.5, and 1.3 is nearer 1.5. -0.4 sits between 0.0 and 0.5,
    // nearer 0.5. Both answers are round-to-nearest, so a wrong lowering (a
    // truncating emulation, say) shows up immediately.
    let input: [f32; 2] = [1.3, -0.4];
    let x = client.create_from_slice(f32::as_bytes(&input));
    let out = client.empty(2 * core::mem::size_of::<f32>());
    unsafe {
        e2m1_roundtrip::launch_unchecked::<Rt>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(x.clone(), 2),
            ArrayArg::from_raw_parts(out.clone(), 2),
        );
    }
    let bytes = client.read_one(out).expect("read the round-tripped pair");
    let got = f32::from_bytes(&bytes);
    println!("\nexecuted f32 -> e2m1 -> f32");
    println!("  1.3  -> {}   (round-to-nearest E2M1 is 1.5)", got[0]);
    println!("  -0.4 -> {}   (round-to-nearest E2M1 is -0.5)", got[1]);
}
