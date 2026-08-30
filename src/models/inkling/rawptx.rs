//! Hand-written PTX kernels through the cubecl runtime -- the assembly level
//! of [`rawcuda`](super::rawcuda), with no C++ in between.
//!
//! JP, 2026-08-30: "I'd rather have us emit raw cuda assembly than write C++
//! ... so could we do just inline cuda assembly and then call that from
//! cubecl?" On NVIDIA the assembly one can WRITE is PTX: SASS, what the GPU
//! executes, has no public assembler, and PTX is what NVRTC hands the driver
//! anyway. So a kernel here is a string of PTX, and it rides exactly the path
//! a hand-written C++ kernel does -- a [`RawCudaKernel`] underneath, the same
//! [`RawArgs`], the same stream, arena buffers, CUDA-graph capture and
//! parameter-patch path -- with one difference in the fork: `compile_kernel`
//! (`cubecl-cuda/src/compute/context.rs`, branch `raw-ptx-kernel` of
//! `../cubecl-graph`) sees a source whose first directive is `.version`,
//! skips NVRTC (which cannot take PTX) and hands the bytes straight to
//! `load_ptx` (`cuModuleLoadData` + `cuModuleGetFunction`), the very call
//! NVRTC's output goes through. The driver's JIT assembles the PTX to SASS
//! for the resident GPU, as it does for every generated kernel.
//!
//! # Where PTX-direct differs from the C++ hatch
//!
//! * **Detection** is textual: past leading whitespace and comments, the
//!   source starts with `.version` ([`is_ptx`], the same rule the fork
//!   applies). [`RawPtxKernel::new`] asserts it, because a PTX text without
//!   its header would otherwise reach NVRTC and fail on its first `.` with a
//!   C++ parse error.
//! * **Kernel id / hashing**: unchanged -- the type name, entry point, an
//!   FNV-1a of the text, the dynamic shared bytes, the cube dim and the mode.
//!   Edit the PTX and the id moves.
//! * **PTX cache** (`CUBECL_COMPILATION_CACHE`): never written for a PTX
//!   kernel. The cache exists to skip NVRTC; there is none to skip, and a hit
//!   would hand `load_ptx` the same bytes the source is. The driver's own JIT
//!   cache (`~/.nv/ComputeCache`) still holds the SASS either way.
//! * **`-lineinfo`**: none. NVRTC put `.loc` directives in its PTX; a
//!   hand-written kernel has whatever `.file`/`.loc` its author wrote, which
//!   is nothing. `cuModuleLoadData` takes no JIT options either.
//! * **`__launch_bounds__(N)`** is the `.maxntid` directive after the entry's
//!   parameter list.
//! * **`__grid_constant__`** has no counterpart and needs none: it only stops
//!   the C++ compiler from copying the struct to local memory when its
//!   address is taken; PTX reads the parameter space with `ld.param`
//!   directly. The runtime's grid-constant support is still required (the
//!   underlying `compile` refuses without it), because it is what makes the
//!   scalar blob a by-value parameter at all instead of a device buffer.
//! * **The toolkit's PTX ISA** matters now. `.version` must be one the
//!   driver's JIT accepts -- see [`PTX_VERSION`].
//!
//! # The PTX-level convention
//!
//! The launch binds the same parameter list as for a C++ kernel (module doc
//! of [`rawcuda`](super::rawcuda)): one device pointer per
//! [`RawArgs::buffer`] in call order, then the packed scalars by value.
//! At the PTX level that is:
//!
//! ```text
//! .version 8.8                       // PTX_VERSION -- ptx_header() emits
//! .target sm_121a                    // PTX_TARGET     these three lines
//! .address_size 64
//!
//! .extern .shared .align 16 .b8 smem[];   // only with with_dynamic_shared
//!
//! .visible .entry <name>(
//!     .param .u64 b0,                // RawArgs::buffer, in call order:
//!     .param .u64 b1,                //   one generic device address each,
//!     ...                            //   at the handle's byte offset
//!     .param .align 8 .b8 info[N]    // RawArgs scalars, BY VALUE, N bytes
//! )
//! .maxntid <x>, <y>, <z>             // == __launch_bounds__; the CubeDim
//! {
//!     .reg .b64 %rd<2>;
//!     .reg .f32 %f<1>;
//!     .reg .b32 %r<1>;
//!
//!     ld.param.u64 %rd0, [b0];       // generic address ...
//!     cvta.to.global.u64 %rd0, %rd0; // ... to global, for ld/st.global
//!     ld.param.f32 %f0, [info];      // scalar at byte 0
//!     ld.param.u32 %r0, [info+4];    // scalar at byte 4
//!     ...
//!     ret;
//! }
//! ```
//!
//! * **Buffers** are `.param .u64`: `cuLaunchKernel` copies eight bytes of
//!   pointer per entry. The value is a generic address (device pointers are
//!   valid generic addresses in the unified address space); `cvta.to.global`
//!   converts it for the `.global` state-space forms, or use generic `ld`/`st`
//!   and skip the conversion. No length, rank, shape or stride travels with a
//!   buffer; pass a length as a scalar.
//! * **Scalars** are ONE byte-array parameter, last. The launch passes a
//!   pointer to the packed blob and the driver copies the DECLARED `N` bytes
//!   from it, so `N` must not exceed the blob: [`RawArgs`] pads the blob to
//!   eight bytes, so `N` = the C-layout size of the scalars rounded up to a
//!   multiple of eight is always exactly the blob (`8 * words().len()`).
//!   Field offsets are the C-layout offsets [`RawArgs`] documents -- each
//!   scalar aligned to its own size -- so `.f32(a).u32(n)` is `info[8]` with
//!   `a` at `[info]` and `n` at `[info+4]`, and `.u32(n).u64(p)` is
//!   `info[16]` with `p` at `[info+8]`. Read them with `ld.param.<type>`.
//!   `.align 8` matches the word packing. With no scalars, omit the
//!   parameter, exactly as for a C++ kernel.
//! * **Comptime / shape specialisation** is the text: bake constants into
//!   the string (`format!`) and the id keys on the bytes.
//! * **Block size.** `.maxntid x, y, z` with the [`CubeDim`] given to
//!   [`RawPtxKernel::new`], which is the launch's block dimensions and part
//!   of the id.
//! * **Shared memory.** Static: `.shared .align 16 .b8 buf[BYTES];` at
//!   module scope (48 KiB static limit). Dynamic: declare
//!   `.extern .shared .align 16 .b8 smem[];` and give the byte count to
//!   [`RawPtxKernel::with_dynamic_shared`]; it becomes the launch's dynamic
//!   shared-memory size and the function's max-dynamic-shared attribute.
//! * **Entry point.** `.visible .entry <name>`; `cuModuleGetFunction` looks
//!   the symbol up by that name, unmangled.

use std::sync::Arc;

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::server::Handle;

use super::rawcuda::{RawArgs, RawCudaKernel};

/// The PTX ISA version every kernel here declares, `.version 8.8`.
///
/// Two constraints meet in this number. It must be HIGH enough for the
/// `.target` and the instructions the kernel uses -- `sm_121a` (GB10) is
/// named from PTX ISA 8.8, which shipped with CUDA 12.9 -- and LOW enough
/// for the driver's JIT, which accepts every ISA up to the one its own
/// toolkit emits. On the Spark, `nvcc --version` reports release 13.0
/// (driver 580.x), whose PTX ISA is 9.0, so 8.8 loads there and on any
/// 12.9+ box. Confirm on the box with
///
/// ```text
/// nvcc --version | grep release          # >= 12.9 is the requirement
/// ptxas -arch=sm_121a -o /dev/null <(cargo run ... scale_ptx)   # or paste it
/// ```
///
/// or simply run the device test below: an ISA the driver refuses fails at
/// `load_ptx` with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, and an ISA too low
/// for the target fails in the same place. A kernel that needs a newer
/// instruction writes its own header instead of [`ptx_header`].
pub const PTX_VERSION: &str = "8.8";

/// The `.target` every kernel here declares: `sm_121a`, the same
/// architecture-specific target NVRTC is given for every generated kernel
/// (`compile_kernel` passes `--gpu-architecture=sm_121a` on a GB10). The `a`
/// suffix unlocks the architecture-specific instructions (the tensor-core
/// and FP4 paths); the module then loads only on that exact architecture,
/// which is the one this crate targets.
pub const PTX_TARGET: &str = "sm_121a";

/// The three directives every PTX module opens with, from [`PTX_VERSION`]
/// and [`PTX_TARGET`]: `.version`, `.target`, `.address_size 64`.
pub fn ptx_header() -> String {
    format!(".version {PTX_VERSION}\n.target {PTX_TARGET}\n.address_size 64\n")
}

/// Whether `source` is a PTX module: past leading whitespace and comments
/// (`//` to end of line, `/* */`), its first token is the `.version`
/// directive, which the PTX ISA requires to open every module. The same rule
/// the fork's `compile_kernel` applies to route around NVRTC -- keep the two
/// in step.
pub fn is_ptx(source: &str) -> bool {
    let mut s = source;
    loop {
        s = s.trim_start();
        if let Some(rest) = s.strip_prefix("//") {
            s = rest.split_once('\n').map_or("", |(_, r)| r);
        } else if let Some(rest) = s.strip_prefix("/*") {
            s = rest.split_once("*/").map_or("", |(_, r)| r);
        } else {
            return s.starts_with(".version");
        }
    }
}

/// A hand-written PTX kernel, launchable on the cubecl CUDA client.
///
/// A [`RawCudaKernel`] whose source is PTX rather than C++: the same id, the
/// same `compile`, the same launch. The type exists so that a call site says
/// which it holds and so that construction checks the text is PTX at all.
#[derive(Clone, Debug)]
pub struct RawPtxKernel {
    inner: RawCudaKernel,
    /// Kept so that the text can be dumped for an offline `ptxas` check.
    ptx: Arc<str>,
}

impl RawPtxKernel {
    /// A kernel from its entry point, its PTX and its block size.
    ///
    /// `name` must be the symbol of a `.visible .entry` in `ptx`; `cube_dim`
    /// is the block size it is launched with and should agree with the
    /// entry's `.maxntid`.
    ///
    /// # Panics
    ///
    /// If `ptx` does not open with `.version` ([`is_ptx`]): such a text
    /// would be handed to NVRTC as C++.
    pub fn new(name: impl Into<String>, ptx: impl Into<String>, cube_dim: CubeDim) -> Self {
        let name = name.into();
        let ptx: Arc<str> = ptx.into().into();
        assert!(
            is_ptx(&ptx),
            "raw PTX kernel `{name}`: the source does not open with `.version`, so the \
             runtime would hand it to NVRTC as CUDA C++; start it with `ptx_header()`"
        );
        Self {
            inner: RawCudaKernel::new(name, ptx.to_string(), cube_dim),
            ptx,
        }
    }

    /// Ask for `bytes` of dynamic shared memory (`.extern .shared`).
    pub fn with_dynamic_shared(mut self, bytes: usize) -> Self {
        self.inner = self.inner.with_dynamic_shared(bytes);
        self
    }

    /// The entry point.
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// The block size, as given to [`RawPtxKernel::new`].
    pub fn cube_dim(&self) -> CubeDim {
        self.inner.cube_dim()
    }

    /// The PTX text, verbatim.
    pub fn ptx(&self) -> &str {
        &self.ptx
    }

    /// Launch over `count` blocks with `args`.
    ///
    /// # Safety
    ///
    /// As [`RawCudaKernel::launch`]: the kernel is hand-written and
    /// unchecked. The caller guarantees that every buffer it reads or writes
    /// is bound and long enough for the indexing the PTX performs, that the
    /// scalars pushed match the `info[N]` bytes the entry declares field for
    /// field, and that the kernel terminates.
    pub unsafe fn launch(
        &self,
        client: &ComputeClient<CudaRuntime>,
        count: CubeCount,
        args: RawArgs,
    ) {
        // SAFETY: forwarded to the caller, who holds the source.
        unsafe { self.inner.launch(client, count, args) }
    }
}

/// The smoke kernel's body: `out[i] = alpha * x[i]` over `n` floats, one
/// thread per element, 256 a block, `info_st { float alpha; unsigned int n; }`
/// -- the whole convention in PTX. [`ptx_header`] goes in front of it.
///
/// The multiply is `mul.rn.f32`: one IEEE round-to-nearest multiply,
/// denormals kept, no contraction -- the same operation NVRTC emits for the
/// C++ `alpha * x[i]` (and for Burn's `mul_scalar`), so the results are
/// bit-identical.
const SCALE_BODY: &str = r#"
// out[i] = alpha * x[i] for i < n. info: { float alpha @0; unsigned int n @4 }
.visible .entry scale_f32(
    .param .u64 x_ptr,
    .param .u64 out_ptr,
    .param .align 8 .b8 info[8]
)
.maxntid 256, 1, 1
{
    .reg .pred %p<1>;
    .reg .b32  %r<5>;
    .reg .f32  %f<3>;
    .reg .b64  %rd<5>;

    // i = ctaid.x * ntid.x + tid.x
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    // tail guard: if (i >= n) return
    ld.param.u32 %r4, [info+4];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    // &x[i], &out[i]
    ld.param.u64 %rd0, [x_ptr];
    ld.param.u64 %rd1, [out_ptr];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mul.wide.u32 %rd2, %r3, 4;
    add.s64 %rd3, %rd0, %rd2;
    add.s64 %rd4, %rd1, %rd2;

    // out[i] = alpha * x[i]
    ld.global.f32 %f0, [%rd3];
    ld.param.f32 %f1, [info];
    mul.rn.f32 %f2, %f1, %f0;
    st.global.f32 [%rd4], %f2;

DONE:
    ret;
}
"#;

/// Threads per block of [`scale_kernel`]; the `.maxntid` in [`SCALE_BODY`].
const SCALE_UNITS: u32 = 256;

/// The smoke kernel's complete PTX: [`ptx_header`] then [`SCALE_BODY`].
pub fn scale_ptx() -> String {
    format!("{}{}", ptx_header(), SCALE_BODY)
}

/// [`scale_ptx`] as a launchable kernel.
pub fn scale_kernel() -> RawPtxKernel {
    RawPtxKernel::new("scale_f32", scale_ptx(), CubeDim::new_1d(SCALE_UNITS))
}

/// `alpha * x` over `n` f32 elements of `x`, into a fresh buffer, through
/// [`scale_kernel`].
///
/// # Safety
///
/// `x` must hold at least `n` f32 values.
pub unsafe fn scale_f32(
    client: &ComputeClient<CudaRuntime>,
    x: &Handle,
    n: usize,
    alpha: f32,
) -> Handle {
    let out = client.empty(n * core::mem::size_of::<f32>());
    let blocks = n.div_ceil(SCALE_UNITS as usize) as u32;
    // SAFETY: `x` holds `n` floats by the caller's contract, `out` was just
    // allocated to hold `n`, and the kernel guards `i < n`.
    unsafe {
        scale_kernel().launch(
            client,
            CubeCount::new_1d(blocks),
            RawArgs::new()
                .buffer(x)
                .buffer(&out)
                .f32(alpha)
                .u32(n as u32),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The detection rule, on the host: a PTX module is recognised through
    /// leading whitespace and both comment forms, and CUDA C++ is not.
    #[test]
    fn ptx_is_told_from_cpp_by_its_first_directive() {
        assert!(is_ptx(".version 8.8\n.target sm_121a"));
        assert!(is_ptx("\n  \t.version 8.8"));
        assert!(is_ptx("// smoke\n// kernel\n.version 8.8"));
        assert!(is_ptx("/* block\n comment */ .version 8.8"));
        assert!(is_ptx("// a\n/* b */\n// c\n.version 8.8"));
        assert!(!is_ptx(""));
        assert!(!is_ptx("// only a comment"));
        assert!(!is_ptx("/* unterminated"));
        assert!(!is_ptx("extern \"C\" __global__ void k() {}"));
        assert!(!is_ptx("#include <cuda_fp16.h>\n.version 8.8"));
        assert!(!is_ptx(".target sm_121a\n.version 8.8"), "must be FIRST");
    }

    /// The smoke kernel's text is PTX, opens with the pinned header and
    /// names its entry; the kernel built from it carries that name and the
    /// block size its `.maxntid` states.
    #[test]
    fn scale_ptx_begins_with_version_and_names_the_entry() {
        let ptx = scale_ptx();
        assert!(is_ptx(&ptx));
        assert!(ptx.starts_with(&format!(".version {PTX_VERSION}\n.target {PTX_TARGET}\n")));
        assert!(ptx.contains(".address_size 64\n"));
        assert!(ptx.contains(".visible .entry scale_f32("));
        assert!(ptx.contains(&format!(".maxntid {SCALE_UNITS}, 1, 1")));

        let k = scale_kernel();
        assert_eq!(k.name(), "scale_f32");
        assert_eq!(k.cube_dim(), CubeDim::new_1d(SCALE_UNITS));
        assert_eq!(k.ptx(), ptx);
    }

    /// The `info[N]` the entry declares is exactly the blob the packer builds
    /// for `.f32(alpha).u32(n)`: the driver copies N bytes from the blob, so
    /// N over the blob's length would read past it.
    #[test]
    fn info_param_size_matches_the_packed_scalars() {
        let ptx = scale_ptx();
        let (_, rest) = ptx
            .split_once(".b8 info[")
            .expect("the entry declares `info[N]`");
        let (n, _) = rest.split_once(']').unwrap();
        let declared: usize = n.parse().unwrap();
        let packed = RawArgs::new().f32(0.7).u32(1000).words().len() * 8;
        assert_eq!(declared, packed, "info[N] must be the blob's byte length");
    }

    /// A text that is not PTX is refused at construction rather than handed
    /// to NVRTC as C++.
    #[test]
    #[should_panic(expected = "does not open with `.version`")]
    fn cpp_text_is_refused() {
        let _ = RawPtxKernel::new(
            "k",
            "extern \"C\" __global__ void k() {}",
            CubeDim::new_1d(64),
        );
    }

    /// The id tracks the text, as for a C++ kernel: two PTX kernels with the
    /// same text share an id and a cache hash; a changed body does not.
    #[test]
    fn kernel_id_tracks_the_ptx() {
        use cubecl::prelude::KernelMetadata;
        let a = scale_kernel();
        let same = scale_kernel();
        let other = RawPtxKernel::new(
            "scale_f32",
            format!("{}// edited\n{}", ptx_header(), SCALE_BODY),
            CubeDim::new_1d(SCALE_UNITS),
        );
        assert_eq!(a.inner.id(), same.inner.id());
        assert_ne!(a.inner.id(), other.inner.id());
        assert_eq!(a.inner.id().stable_hash(), same.inner.id().stable_hash());
        assert_ne!(a.inner.id().stable_hash(), other.inner.id().stable_hash());
    }

    /// Deterministic values in `[-1, 1)`, the same bytes on every machine.
    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The hand-written PTX against the eager path -- Burn's `mul_scalar` on
    /// the same CUDA client -- on `n = 1000` floats, a length that is not a
    /// multiple of the block so the tail guard is exercised. Both sides
    /// perform one IEEE round-to-nearest f32 multiply (`mul.rn.f32` here),
    /// so the results are BIT-IDENTICAL and the assertion is exact.
    ///
    /// Launched TWICE with two alphas: the second launch has the same kernel
    /// id as the first, so it hits the loaded module and moves only the
    /// by-value blob -- the property the CUDA-graph patch path relies on.
    ///
    /// NOT EXECUTED on the machine this was written on (no CUDA device; `Bk`
    /// is `burn::backend::Cuda`). On a Spark, with `../cubecl-graph` on
    /// branch `raw-ptx-kernel`:
    /// `cargo test --release --features inkling-cuda --lib models::inkling::rawptx -- --nocapture`
    #[test]
    fn scale_matches_the_eager_path() {
        use super::super::seam::{Bk, client_of, handle_of, tensor_of};
        use burn::tensor::{Tensor, TensorData};

        let dev = Default::default();
        let n = 1000usize;
        let xv = lcg(n, 3);
        let x: Tensor<Bk, 1> = Tensor::from_data(TensorData::new(xv, [n]), &dev);
        let client = client_of(&x);
        let xh = handle_of(x.clone());

        for alpha in [0.7f32, -3.25] {
            let want = x
                .clone()
                .mul_scalar(alpha)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            // SAFETY: `xh` holds `n` floats.
            let out = unsafe { scale_f32(&client, &xh, n, alpha) };
            let got = tensor_of(client.clone(), dev.clone(), out, 1, n)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "alpha {alpha}, element {i}: ptx {g} vs burn {w}"
                );
            }
        }
    }
}
