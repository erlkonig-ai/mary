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
//!
//! ## What this GB10 answered, 2026-08-25
//!
//! Reported (this binary, on the Spark): `e2m1` = `Conversion`; `e2m1x2`,
//! `e4m3`, `e5m2`, `ue8m0`, `e2m3`, `e3m2` = `Conversion | Buffer`. Executed:
//! `1.3 -> 1.5`, `-0.4 -> -0.5`, i.e. round-to-nearest, not truncation.
//!
//! Underneath, `cubecl-cuda`'s `register_type_usage` gates FP4/FP6 on
//! `arch_major` in `{10, 11, 12}`; sm_121 is major 12, so the capability is
//! not a Blackwell-datacentre feature that happens to be reported here — it is
//! registered for this family deliberately.
//!
//! ### The instruction is real, and the `a` suffix is load-bearing
//!
//! `cvt.rn.satfinite.e2m1x2.f32` assembles for `sm_121a` and becomes one SASS
//! instruction, `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` — two f32 in, one
//! byte holding two E2M1 codes out. It does *not* assemble for plain `sm_121`
//! ("Instruction 'cvt with .e2m1x2' not supported"), and `cuda_fp4.hpp` gates
//! its inline-PTX path on `__CUDA_ARCH_FAMILY_SPECIFIC__`, which `-arch=sm_121`
//! leaves undefined; the header then falls back to an emulation that is
//! *worse* than the software ladder (88 vs 24 SASS instructions for one
//! conversion). CubeCL passes `--gpu-architecture=sm_{arch}a` whenever
//! `arch.version >= 90`, and `sm_121a` defines both `__CUDA_ARCH_SPECIFIC__`
//! and `__CUDA_ARCH_FAMILY_SPECIFIC__`, so CubeCL-generated kernels get the
//! hardware path. `rn` is the only rounding modifier the instruction accepts,
//! and `satfinite` is mandatory.
//!
//! ### What the hardware does *not* offer
//!
//! Nothing helps with the per-16 block amax. On `sm_121a` `ptxas` rejects
//! `redux.sync.max.f32` (and `.abs`, `.NaN`) with "Instruction 'redux.f32' not
//! supported" — that is the Blackwell-datacentre float warp reduction, and it
//! is absent here, as is `max.f32x2`, `atom.global.max.f32`, and every
//! `tcgen05.*`. Only integer `redux.sync.max.u32` exists (`REDUX.MAX`), usable
//! for an amax only via the abs-bit-pattern trick and only in a
//! one-thread-per-element layout this kernel does not have. There is no
//! quantize/scale-generation opcode of any kind: `blockscale` appears in
//! `ptxas` solely as a modifier of the block-scaled MMA that *consumes* a
//! scale. The E4M3 scale byte is already one `F2FP.SATFINITE.E4M3`.
//!
//! ### Measured against the kernel itself
//!
//! Framing rule: static SASS instruction count for one thread = one 16-element
//! NVFP4 block, `NOP`/`BRA` excluded; source is the CUDA CubeCL itself
//! generated for `fp4quant::quantize_nvfp4_kernel<f32>` (captured by running
//! this binary under `CUBECL_DEBUG_OPTION=source`), assembled with CUDA 13.0
//! V13.0.88 `nvcc --gpu-architecture=sm_121a -lineinfo -cubin`. Static count,
//! not wall time — no timing was taken, the box was busy.
//!
//! | variant                                            | instr | regs |
//! | -------------------------------------------------- | ----- | ---- |
//! | as generated today                                 |   747 |   33 |
//! | + hoisted reciprocal, `__nv_cvt_float_to_fp4`      |   397 |   32 |
//! | + amax as `fmaxf(a, fabsf(v))` instead of two `if` |   350 |   32 |
//!
//! The 397 line replaces 17 `FCHK`+`CALL` pairs (the full-precision divide
//! slow path — CubeCL emits `x / scale` per element *and* `amax / 6.0`) and
//! 130 of 166 `FSETP` plus 110 of 115 `SEL` (the seven-midpoint ladder) with
//! 16 more `F2FP`. The 350 line is not a hardware win at all: it is the amax
//! written as a max rather than a compare-and-select, worth 47 instructions.
//! A packed `__nv_cvt_float2_to_fp4x2` would fold the 16 `F2FP` into 8 and
//! needs CubeCL's `Vector<e2m1x2, N>`; the 16 scalar `LDG` are untouched by
//! any of this and are the obvious next lever.
//!
//! ### The tie difference, measured
//!
//! Hardware `cvt.rn` is round-to-nearest **ties-to-even in code space**; the
//! ladder's `>=` sends a tie **away from zero**. They part on exactly four of
//! the seven midpoints (`0.25 -> 0` vs `1`, `1.25 -> 2` vs `3`, `2.5 -> 4` vs
//! `5`, `5.0 -> 6` vs `7`); on `0.75`, `1.75` and `3.5` rounding up *is* the
//! even code and they agree. Hardware also keeps the sign of `-0.0` (code
//! `0x8`, where the ladder gives `0x0`) and saturates NaN to `±6.0`.
//!
//! Running both kernels over 1,048,576 elements: on continuous
//! activation-shaped data (heavy-tailed, `u * exp(...)`) **zero** codes and
//! zero scale bytes differ — exact midpoints are measure-zero for such inputs.
//! On data deliberately drawn from a coarse dyadic grid, so quotients land on
//! midpoints often, 2.211% of codes differ, every one of them by exactly one
//! E2M1 step, and the scale byte never differs.

use cubecl::features::TypeUsage;
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

    // Launch the production quantizer once, so a `CUBECL_DEBUG_OPTION=source`
    // run of this binary emits the CUDA that `fp4quant` actually compiles.
    // Reading the re-creation of a kernel is not reading the kernel.
    let rows = 1usize;
    let k = 64usize;
    let dense: Vec<f32> = (0..rows * k).map(|i| (i as f32) * 0.01 - 0.3).collect();
    let xh = client.create_from_slice(f32::as_bytes(&dense));
    let (codes, scales) = mary::models::inkling::fp4quant::quantize_nvfp4(&client, &xh, rows, k);
    let cb = client.read_one(codes).expect("read the codes");
    let sb = client.read_one(scales).expect("read the scale bytes");
    println!(
        "\nfp4quant::quantize_nvfp4 on {rows}x{k}: {} code words, {} scale bytes",
        cb.len() / 4,
        sb.len()
    );
}
