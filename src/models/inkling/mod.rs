//! Inkling (Thinking Machines) — a 42-layer (276 B / 12 B active) and 66-layer
//! (975 B / 41 B active) sparse-MoE decoder with native audio and vision input.
//!
//! What is here so far: the configuration and the checkpoint-name to
//! module-slot layout, gated as a bijection against a real checkpoint by
//! `inkling_layout_gate`.
//!
//! Relative to Kimi-K3 the attention is conventional — GQA, not KDA plus MLA —
//! but the block adds a depthwise short convolution on the attention input, on
//! K, on V and into the MLP; QK-norm; and a rank-16 learned relative-position
//! logit path in place of RoPE. The MoE block is close enough to K3's
//! sigmoid router with shared experts and gate bias to port with parameter
//! changes. The FP4 decode is *not*: K3 is MXFP4 (E8M0 scales, block 32),
//! Inkling is NVFP4 (E4M3 scales, block 16, plus a per-expert F32 second level).

// Nothing here is conditional any more. The module tree used to be gated three
// ways — `inkling` for the headers, `inkling-burn` for the Burn lane,
// `inkling-cuda` for the device one — and the gating was already a fiction:
// `fp4gemm` below sat outside all of it and needs cubecl, so the header-only
// build had not compiled in some time. One feature, one lane, no cfg.
pub mod attn;
pub mod block;
pub mod burn;
pub mod config;
// The NVFP4 ACTIVATION quantiser, on the device. The routed-expert lane calls
// it twice per expert; there is no host twin in the data plane to select
// between, so there is nothing to gate.
pub mod fp4quant;
pub mod layer;
pub mod layout;
pub mod load;
// The DENSE MLP on the host, and only that. The f32 MoE reference that used to
// be the bulk of this file -- `routed_experts`, `shared_experts`, `moe`,
// `expert_ffn_one` -- is deleted: it had no caller in the data plane and it was
// the readable version of the hottest function in the model, which is the
// combination that gets a dead lane optimised instead of the live one. What is
// left is what the MTP heads run.
pub mod mlp;
pub mod mtp;
pub mod nvfp4;
pub mod pile;
// Where a Burn tensor and a raw cubecl handle are admitted to be the same
// bytes. Two functions; it is what lets the residual stream stay on the device
// across a lane boundary that is a dialect boundary and nothing more.
pub mod seam;
// Which memory pool a prefill wants, and when to hand its pages back. The
// per-buffer cap is `budget`; this is the other thing the allocator does, which
// is to reserve three to five times what the prefill holds.
pub mod pool;
// The residual stream: the switch that decides its dtype, and the two kernels
// -- the residual add and RMS normalization -- that let it be BF16 without
// widening back at every seam. It was the last wide storage in an otherwise
// narrow layer: 48 KiB a token across its three f32 buffers, halved at BF16.
pub mod resid;
// The QK-norm as one kernel per operand: RMS over `head_dim` with the width a
// compile-time constant and both passes unrolled, in place of the six Burn
// launches (square, mean, add, sqrt, divide, gain) each norm cost -- twelve a
// layer in a decode step. Behind `INK_HEAD_RMS_NATIVE`, off until measured.
pub mod headnorm;
// Hand-written CUDA C++ kernels through the SAME cubecl runtime -- stream,
// arena, graph capture and patch, by-value scalars, PTX cache -- with no second
// runtime beside it. A `CubeTask` whose `compile` returns the source verbatim,
// and the argument convention such a kernel is written against.
pub mod rawcuda;
// The same hatch one level down: hand-written PTX, the assembly NVRTC would
// have emitted, handed to the driver as it is. Same `RawArgs`, same runtime;
// the fork routes a `.version`-led source around NVRTC (`raw-ptx-kernel`).
pub mod rawptx;
// The routed lane's NVFP4 GEMM as hand-written PTX -- `fp4_linear_swz`'s
// operands, tile decomposition, lane roles and output layout, with the K loop
// unrolled into the text and every address a base plus an immediate. Generated
// per shape, so the dims are literals and the kernel id keys on them.
pub mod ptxgemm;
// The short convolution's decode step, as one kernel instead of nineteen Burn
// ops. Four run per layer and they were a third of every launch in a decode
// step; the arithmetic is 16384 multiply-adds.
pub mod sconv;
// The attention score epilogue -- scale, relative-position bias, causal mask
// and sliding window -- as one kernel over `[heads, n, n]` instead of five
// materialised tensors of that shape and two host loops over `n^2`.
pub mod scorebias;
// Sliding-window attention as a BAND: one kernel per (head, query) over the
// <= 512 keys the window admits, with no `[heads, n, n]` anywhere. Thirty-five
// of the forty-two attention layers are local, and they were computing the
// full n^2 and masking it away.
pub mod banded;
// The OTHER seven layers: global attention fused the same way, with an online
// softmax so the key axis is never materialised either. The band's trick does
// not transfer -- a global layer really does read every key -- so this one
// tiles the key axis and carries the softmax's running max and sum across the
// tiles, which is the one part flash attention genuinely invented.
pub mod flash;
// What one attention layer asks the allocator for, and whether this device
// will give it: the `[heads, n, n]` score matrix against the per-buffer cap
// cubecl sets at `cuDeviceTotalMem / 4`.
pub mod budget;
// A panic on any thread ends the process. Without it, a refused allocation
// panics a cubecl worker and the run exits 0 with a plausible wrong answer.
pub mod fatal;
// Per-pass system state for the intermittent multi-second decode stall: the
// host CPU, faults, huge pages and pressure that a timer inside the process
// cannot see. Off unless `INK_STEPSTAT=1`.
pub mod stepstat;
// The MMA's M padding, written by the kernel that was already producing the
// buffer instead of by a `zeros` and a `cat` beside it.
pub mod pad;
// One interface over the two places a running model's weights can come from —
// a safetensors checkpoint or a pile — plus the residency cache and the byte
// counters, which belong to the asking rather than to either storage.
/// A paged store for one layer's keys or values — see the module docs for
/// why pages rather than a contiguous tensor.
pub mod kvpages;
pub mod source;
// Token-TREE speculative decoding over the chained MTP heads: topology,
// top-b, the drafting schedule, the ancestor masks and the accept walk. Pure
// index algebra, no backend — so the parts that are easy to get subtly wrong
// are tested without a GPU.
pub mod spectree;
pub mod stack;
// The scalar and cross-layer transaction law for a speculative target pass.
// Pure bookkeeping: the CUDA Session supplies caches through its tiny trait,
// while the accept/abort invariants remain testable without a model or device.
pub(crate) mod target;
pub mod vision;

// THE ASSEMBLY, and then the thing it assembles.
//
// Every module above is a COMPONENT. None of them says how a model is built out
// of them, and until 2026-08-27 nothing in the library did: the composition
// lived in `src/bin/inkling_forward.rs`, and nothing can link against a binary.
// So a long-lived process could not hold this model — every request meant
// launching a program, and 144 GiB of weights meant that was never going to be
// a serving path.
//
// `assembly` is that composition, moved verbatim out of the binary. `session`
// is the handle it makes possible: weights, KV and position as ONE value that
// survives across calls.
pub mod assembly;
pub mod session;

// …and the thing that makes a held model reachable by another PROGRAM. A
// `Session` lives in one address space, and the program that wants it — `drive`
// — must not link mary. So the model runs in its own process and is reached
// over the framed-stream convention. The module is backend-free (it is a
// protocol and a client), which is why it is also compiled without the CUDA
// lane; see the shim in `models/mod.rs`.
#[cfg(feature = "serve")]
pub mod serve;
// Durable observability is native graph exhaust. The serving wire remains free
// to use JSON framing, but a resident Drive mind drains these typed facts after
// startup and after every turn; there is no separate export lifecycle.
#[cfg(feature = "drive-mind")]
pub mod telemetry;

pub use config::InklingConfig;
// Layer 2's experts are BF16 and have no scales; the same tiling, the same
// device residency, the unscaled sibling of the instruction.
pub mod bf16gemm;
pub mod fp4gemm;
// The half of NVFP4 that costs no calibration: the WEIGHT stays four bits and
// the activation stays BF16, dequantised in registers inside the B-fragment
// load. For every tensor the publisher left BF16 and never gave an
// `input_quantizer` -- the sink experts, the attention projections, and the
// 1.65 GB unembedding.
pub mod w4a16gemm;
// The head as an approximate maximum-inner-product search rather than an
// exhaustive one: a 1-bit sign sketch in a rotated basis, scanned at a tenth of
// the NVFP4 table's bytes, and an exact rescore of the shortlist it names.
pub mod annhead;
// The same lane as `fp4gemm`, but the loop over a layer's active experts moved
// off the host and into the grid: one launch per stage per layer, with the
// block index selecting the expert. Same kernels, same order, same bits.
pub mod moegroup;
// The router DECISION -- sigmoid, top-k, log-softmax -- on the device. The
// projection was already there; this is the 66 ms of host sort that sat between
// the readback and the expert launch at prefill.
pub mod routetopk;

// WHICH SLICE of each tensor a rank owns, for the split that cuts WITHIN a
// layer rather than between layers. `INK_LAYERS` divides where the bytes live;
// this divides how many bytes a token has to traverse, which at batch one is
// the difference between `M / B` and `M / (2B)`. The header carries the
// interconnect arithmetic that decides whether the per-layer all-reduce it
// costs is affordable -- on this fabric it is 2.54 ms against a measured
// 105.2 ms token, so it is not close.
pub mod tp;

// THE COLLECTIVE that puts `tp`'s slices back together, and the rendezvous that
// forms the NCCL group across two BOXES rather than two devices in one process.
// Split from `tp` because `tp` is arithmetic and its tests run anywhere, while
// nothing here runs without two GPUs and a wire. The property its header
// defends is not the 29.56 us latency but the fact that the collective is
// STREAM-ORDERED and never blocks the calling thread: a collective that synced
// the host would serialise the layer loop 84 times a token and turn
// `max(enqueue, device)` back into `enqueue + device`, which is the whole win.
pub mod tpcomm;

// `tp`'s index ranges turned into BYTES, and which of them still alias the pile.
// The arithmetic is symmetric between an output-axis cut and an input-axis cut;
// the COST is not. A row range is a span of the mapping and binds for free; a
// column range is a stride and must be gathered into a fresh allocation the
// pile does not back. Every function here can be got wrong in a way that yields
// finite numbers and fluent text, so each carries a test pinning the exact
// wrong answer as well as the right one.
pub mod tpshard;

// The routed-expert ROW PLAN on the device, from a top-k answer that is never
// read back. `routetopk` moved the decision; this moves everything downstream
// of it that the host used to need the decision's VALUE for.
pub mod devplan;
