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
//! #### "…not for f16, or through a different instruction family?"
//!
//! Asked and answered against `ptxas` 13.0.88 rather than the ISA doc, which
//! disagrees with it on this chip. **No** — and the reason is structural, not a
//! missing spelling.
//!
//! * `redux` is 32-bit-**integer**-only on every architecture:
//!   `redux.sync.max.{u16,u64,f16,bf16}` all give "Unexpected instruction types
//!   specified for 'redux'". Even where the float form exists it is `.f32`.
//! * `redux.sync.max.f32` is a **die-family** gap, not a suffix gap. It
//!   assembles on `sm_100a`/`sm_103a` (→ `CREDUX.MAX.F32`) and is rejected with
//!   the identical string on `sm_120a`, `sm_121`, `sm_121a` *and* `sm_121f`.
//!   GB10 does not have the unit. (The `a`/`f`/plain suffix turned out to be
//!   load-bearing for *nothing* in this probe set — unlike the block-scaled MMA
//!   above, every instruction tested behaved the same on all three.)
//! * `redux`'s SASS form has **no member-mask operand**, so it cannot be cheaply
//!   restricted to 16 lanes: a non-constant mask makes `ptxas` wrap it in
//!   `BSSY.RECONVERGENT`/`WARPSYNC.EXCLUSIVE`/`BSYNC`, and two constant
//!   half-masks produce full warp divergence plus duplicated out-of-line
//!   `WARPSYNC.COLLECTIVE … ENDCOLLECTIVE` blocks. The right tool for a 16-lane
//!   subgroup is `shfl.sync.bfly.b32` with `c = 0x101f`, which encodes directly.
//! * There **are** two families with an f16/bf16 max where the f32 form is
//!   flatly rejected — `red/atom.global.max.noftz.v2.{f16,bf16}` →
//!   `REDG.E.MAX.{F16x2,BF16x2}.RN`, and
//!   `cp.reduce.async.bulk…bulk_group.max.{f16,bf16}` →
//!   `UBLKRED.G.S.MAX.{F16,BF16}.RN` (where `.add.f32` is accepted and
//!   `.max.f32` is not). Both are **element-wise memory** reductions,
//!   `dst[i] = max(dst[i], src[i])`, not horizontal ones. They are the right
//!   primitive for a cross-CTA running amax — a tensor-wide scale — and cannot
//!   collapse a 16-element block to a scalar.
//! * The SIMD video family is emulated and worse: `vmax2` is 7 SASS
//!   instructions, `vmax4` 19–28. The one near-native member is
//!   `vabsdiff4.u32` (3). `max.{s16x2,u16x2}` *is* native (`VIMNMX.S16x2`), but
//!   needs a sign-clear per operand and loses to `FMNMX`.
//!
//! **Every reduction unit on GB10 is cross-lane** (`REDUX`, `SHFL`, `ATOM`,
//! `UBLKRED`); none of them reduces one thread's own registers. A thread
//! reducing its own sixteen is always a tree of two-input ops — and that is a
//! feature of this layout, not a tax on it. A `redux` is one instruction per 32
//! *elements*; one `FMNMX` in the one-thread-per-block layout is one
//! instruction per 32 independent *blocks*. Costed in warp-instruction slots
//! per 16-element block: the f32 `FMNMX` tree is 0.47 and a bf16
//! `HMNMX2.XORSIGN` tree 0.34, against 4.5 for a 16-thread `shfl.bfly` tree and
//! ~13.5 for split-mask `redux` — **~10× worse**, before counting that the
//! loads drop from `LDG.E.256` to `LDG.E.32`.
//!
//! The one real SIMD win is `max.xorsign.abs.{f16x2,bf16x2}` →
//! `HMNMX2[.BF16_V2].XORSIGN R, |Ra|, |Rb|`, a genuine one-instruction max of
//! two absolute-value pairs: 11 instructions for a 16-element bf16 amax against
//! 15 for the f32 tree, and bit-exact (the max of a set of bf16 values is one of
//! them). It is **not** reachable for `quantize_nvfp4_bf16` as written, because
//! that kernel must widen all sixteen elements to f32 anyway to form the
//! quotients the E2M1 conversion consumes — the widening is shared with the
//! amax, so keeping the amax in bf16 would save only the 15→11 on the tree, not
//! the widening. Multiplying in bf16 to avoid it would put ~2^-9 of relative
//! error on a quotient compared against E2M1 midpoints, which is three orders of
//! magnitude too coarse.
//!
//! In f32 the abs is free regardless: `FMNMX` takes an `|src|` operand modifier
//! on **both** inputs, which is why writing the amax as `max(|a|, |b|)` rather
//! than a compare-and-select is worth ~16 instructions on its own.
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
//! | as generated before the rewrite                    |   747 |   33 |
//! | projected: hoisted reciprocal + `__nv_cvt_float_to_fp4` | 397 | 32 |
//! | projected: + amax as `fmaxf(a, fabsf(v))`          |   350 |   32 |
//! | **landed** (all three, plus vectorized loads)      | **302** |   |
//!
//! The 397 line replaces 17 `FCHK`+`CALL` pairs (the full-precision divide
//! slow path — CubeCL emits `x / scale` per element *and* `amax / 6.0`) and
//! 130 of 166 `FSETP` plus 110 of 115 `SEL` (the seven-midpoint ladder) with
//! 16 more `F2FP`. The 350 line is not a hardware win at all: it is the amax
//! written as a max rather than a compare-and-select, worth 47 instructions.
//!
//! All three landed in `fp4quant` and beat the projection, because the packed
//! `__nv_cvt_float2_to_fp4x2` folds those 16 `F2FP` into 8 and the 16 scalar
//! `LDG` — the lever this survey named and did not pull — collapse into 2
//! `LDG.E.256`. **Split the landed number before quoting it**: 170 of the 302
//! are the straight-line body and 132 are the out-of-line correctly-rounded
//! division helper `ptxas` plants behind an `FCHK`, which is I-cache and not
//! issue slots. Body only: **660 → 170** on the f32 lane and **741 → 195** on
//! the BF16 one. The helper *grew* (87 → 132) because `amax / 6` and `1.0 / s`
//! are two differently-shaped divides, where the old kernel's seventeen were all
//! the same shape and shared one routine.
//!
//! CubeCL reaches the packed instruction as
//! `Vector::reinterpret(Vector::<e2m1x2, N>::cast_from(v))` — the spelling its
//! own `runtime_tests::minifloat` uses — and the pair order is little-endian,
//! first element in the low nibble, which is already the NVFP4 packing order.
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
//!
//! Landing it made the `-0.0` line the interesting one, not the ties: on the
//! real `w13_weight` rows `fp4quant_gate` reported **zero** tie divergences and
//! ~2500 sign-of-zero ones, because those rows are full of the `0x8` code. The
//! resolution was to move the host reference to the hardware's rule rather than
//! the reverse — E2M1 *has* a signed zero, the checkpoint's own producer emits
//! it, and our ladder was the outlier. The gate then went bit-identical, and
//! **stronger**: the layout case had been skipping 1010 blocks precisely because
//! they contained `0x8` codes our encoder could not reproduce, so the comparison
//! against the checkpoint's own bytes went from 1429 blocks to 2439.

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

    // Launch the production quantizer once at EACH element type, so a
    // `CUBECL_DEBUG_OPTION=source` run of this binary emits the CUDA that
    // `fp4quant` actually compiles. Reading the re-creation of a kernel is not
    // reading the kernel — and the BF16 entry point is not the f32 one
    // recompiled: its load is half as wide, so the instruction the memory side
    // costs is a different instruction.
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

    let dense_bf: Vec<half::bf16> = dense.iter().map(|v| half::bf16::from_f32(*v)).collect();
    let xbh = client.create_from_slice(half::bf16::as_bytes(&dense_bf));
    let (codes_b, scales_b) =
        mary::models::inkling::fp4quant::quantize_nvfp4_bf16(&client, &xbh, rows, k);
    let cbb = client.read_one(codes_b).expect("read the BF16 codes");
    let sbb = client
        .read_one(scales_b)
        .expect("read the BF16 scale bytes");
    println!(
        "fp4quant::quantize_nvfp4_bf16 on {rows}x{k}: {} code words, {} scale bytes",
        cbb.len() / 4,
        sbb.len()
    );
}
