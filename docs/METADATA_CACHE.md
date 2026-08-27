# The per-launch metadata upload, and the cache that would remove it

A design note, not an implementation. It exists because the thing it describes
is upstream of three separate workarounds that are already in the tree or were
about to be, and a change that removes the *reason* for three workarounds is
worth more than any of them. Everything numeric here is measured; the framing
rule is attached to each figure.

## What happens today

`cubecl-cuda`'s `CudaServer::launch_checked` splits a launch's packed argument
blob (`bindings.info.data`) in two at `dynamic_metadata_offset`:

* below the offset sit the scalars and the **static** metadata — per-buffer
  lengths, ranks, and the offsets into the second half. This rides to the kernel
  as a by-value `__grid_constant__` struct, so `cuGraphExecKernelNodeSetParams`
  can move it.
* at or above the offset sit **every bound tensor's shape and stride list**.
  These are variable-length, so they cannot be a fixed-size kernel parameter.
  `launch_checked` calls `create_with_data(dyn_meta)`, which reserves a pinned
  host buffer, copies the shapes and strides into it, allocates a device buffer,
  and issues `cuMemcpyHtoDAsync`.

That happens **on every launch that binds a ranked tensor, every time**, whether
or not the shapes have changed since the last launch of the same kernel with the
same bindings.

## What it costs, measured

Framing: per decode step, per node, `INK_LAYERS=0:21`, `INK_KV=1`, ctx3732,
one GB10, `mary`'s inkling decode path.

* **483 host-to-device memcpy nodes**, all 483 from host memory, out of 1783
  kernel launches — 27% of launches carry one.
* **19,280 bytes** total per step. Size histogram, identical between two
  captures of consecutive steps: 16 B (x306), 32 B (x103), 64 B (x57),
  144 B (x2), 208 B (x36).
* **~1869 allocator reservations** per decode step per node (`GRAPHARENA`),
  against 1783 launches — roughly one per launch. Each is a bucket search plus
  bookkeeping, all host work.
* **~6 host stalls per step**, at `cuEventSynchronize`, spread through the layer
  loop at offsets 5.4, 12.3, 17.2, 22.6, 28.9 and 34.1 ms, each 2.7–4.9 ms,
  totalling ~22.7 ms of a ~37 ms bracket. (Anchored nsys, median over the last
  12 warm intervals, timeline cut at `ann_scan_kernel`.) These are cubecl's drop
  queue: `Command::kernel` flushes when 64 staged allocations have accumulated,
  and `PendingDropQueue::flush` syncs the fence from the *previous* cycle.
  483 uploads / 64 per flush = 7.5, against the six the profile found.

The shapes and strides these uploads carry are **constant for 128-step epochs**
at a fixed decode config. The KV pages are pre-allocated at `PAGE = 128` rows
and written in place, and `Pages::parts` hands pages over whole rather than
slicing to the live row count precisely so that a chunk's shape does not walk
`1..=PAGE` as the window advances. So on 127 of every 128 steps every one of the
483 uploads copies **the same bytes it copied last step, from a different host
address, into the same device address**.

## The change

Cache the dynamic metadata. Key it on the binding set — the (rank, shape,
stride) tuples the launch is about to describe — and when the key matches the
one a cached device buffer already holds, bind that buffer and skip the reserve,
the copy and the upload entirely.

Where it goes: `CudaServer::launch_checked`, in the `if grid_constants` arm that
currently reads

```rust
if info.dynamic_metadata_offset < info.data.len() {
    let dyn_meta = &bytemuck::cast_slice(&info.data[info.dynamic_metadata_offset..]);
    handle = Some(command.create_with_data(dyn_meta)?);
}
```

The cache is per stream (device buffers are stream-scoped in this runtime) and
wants a small map from a hash of the dynamic half to a `Handle`, with the full
bytes kept alongside so a hash collision is caught rather than trusted.

## What it removes

1. **The 483 memcpy nodes** disappear from a captured region. That is the
   blocker that shaped the entire cross-step graph design: "0 of 483 whole
   specs shared between two captures" was the headline number, and it measured
   that the pinned SOURCE address moved — which it does unconditionally, once
   per capture — while never measuring whether a byte changed. With no upload
   there is no node and no question.
2. **The drop-queue stalls.** Nothing is staged, so `should_flush` never fires
   and the fence sync never happens. This is a cure rather than a mitigation:
   `CUBECL_DROP_FLUSH_COUNT` (cubecl-graph 1dd9dfec) raises the threshold, which
   past a certain depth also drives the wait to approximately zero, but it does
   so by holding more pinned memory rather than by removing the reason.
3. **The stable-staging-slot idea**, which was the other candidate cure: give
   every launch site a pinned slot it reuses forever, ~19 KB of payload. A
   metadata cache subsumes it, because a launch that does not upload needs no
   slot at all.
4. **Most of the ~1869 reservations.** 483 of them are the pinned staging and
   483 are the device metadata buffers; both go.

## What it costs

* One device buffer per distinct metadata payload, held for the life of the
  cache. At the observed histogram the *whole step's* payload is 19,280 bytes,
  so the distinct set is smaller than that and is measured in kilobytes. The
  earlier census found only **29 distinct destination addresses** across all 483
  uploads with the capture arena on, which is a strong hint that the distinct
  payload count is of that order rather than of order 483.
* A hash of the dynamic half per launch, on the host, in place of a pinned
  reserve plus a memcpy plus a driver call. The dynamic half is 16–208 bytes,
  so the hash is a handful of cycles against work that is currently three
  orders of magnitude more.
* An invalidation obligation: the cached device buffer must not be freed or
  reused while a graph node points at it. Under the capture arena this is
  already the discipline; outside it, the cache holding the `Handle` is enough.

## What has to be checked before it ships

* **That the payload really is invariant**, not merely observed to be. The
  `INK_GRAPH_DIFF` report now splits a moved word on `dyn_offset` and prints how
  many moved words are staged shapes/strides versus by-value scalars, plus a
  per-kernel breakdown. That report answers this directly and should be run
  before anyone writes the cache.
* **Prefill, not just decode.** A prefill's shapes move with the sequence, so
  the cache will miss more often there. It must degrade to today's behaviour on
  a miss, not to a stall.
* **Whether the hash can collide meaningfully.** Keep the bytes and compare them
  on a hit; the payload is at most 208 bytes and the comparison is cheaper than
  the upload it replaces.

## Provenance

The 483 census, the size histogram and the destination-sharing figure are from
`INK_GRAPH_DIFF=1` runs recorded in mary commits `4c69db9`, `d2f903f` and the
`graph-rebase-r3` merge. The 22.7 ms stall figure and the 81%-device-busy step
decomposition are from the anchored nsys instrument (`scripts/nsys-bracket.py`
as rebuilt in `cde5de9`); the earlier differencing method that this file's
figures were NOT taken from was withdrawn on 2026-08-27 as unreliable on this
config. The drop-queue mechanism is read off
`cubecl-runtime/src/memory_management/drop_queue/queue.rs`.

## What happened when it was built

Built on 2026-08-27 as `cubecl-graph` branch `metadata-cache`. Default OFF;
`CUBECL_META_CACHE=1` serves the eager path and is bypassed inside a capture,
`=2` additionally lets a launch inside a capture HIT (it never inserts while a
capture or an arena window is open, which is a correctness requirement: inside a
capture the copy that fills a fresh buffer is a graph NODE and has not run, so
caching it would let a later eager launch read uninitialised memory).

Every figure below is from ONE binary, both arms, on one GB10. Where a number is
a time, the instrument and the resolution are beside it.

### The uploads, counted device-side

Per warm decode step, one node, `INK_LAYERS=0:21`, `INK_KV=1`, ctx3732, anchored
nsys (`ann_scan_kernel`, median over the last 12 warm anchor intervals), one
profiled run per arm, via `scripts/nsys-memcpy-census.py`:

| | `=0` | `=1` |
|---|---|---|
| kernel launches | 1814 | 1814 |
| HtoD memcpy ops | 508 | **3** |
| HtoD bytes | 40,124 | 20,492 |
| summed HtoD device time | 493.5 us (1.09% of the step) | 1.3 us (0.00%) |

The 505 that went away are 16 B x307, 32 B x103, 64 B x57, 144 B x2, 208 B x36 --
this file's own census, reproduced to the count on a second night by a different
instrument. The kernel count is identical, which is the check that the cache
changes what a launch *carries* and not what it *is*.

### Where the step time went

Same profiles, `scripts/nsys-bracket.py`, per decode step:

| | `=0` | `=1` |
|---|---|---|
| step period | 45.075 ms | 41.411 ms |
| device busy | 36.283 ms | 36.227 ms |
| device idle | 8.792 ms | 5.184 ms |
| launcher-thread BLOCKING waits | 25.761 ms | **0.000 ms** |

DEVICE BUSY IS UNCHANGED. The whole of the 3.66 ms is device IDLE removed, and
the mechanism is the one this file predicted: with nothing staged,
`should_flush` never fires, so the drop queue never syncs the previous cycle's
fence, so the 25.8 ms of `cuEventSynchronize` on the launching thread goes to
zero. That is a cure and not a mitigation.

Independently, from the binary's own `pass_ms` (whole decode pass, wall clock),
3 interleaved reps per arm with the leading arm alternated, median over the last
32 of each run's 64 steps: `=0` 44.1 / 44.3 / 44.45 ms, `=1` 40.9 / 42.0 /
40.7 ms. That is **3.08 ms, 7.0%, +/- 1.5 percentage points at 2 sem** -- an
interval that CONTAINS the graph lane's measured 5.9%, so this data does not
distinguish the two and no claim to beat the lane is made from it.

### What a hit costs against the upload it replaces

`CUBECL_TIME_META_CACHE=1`, per LAUNCH THAT BINDS A RANKED TENSOR, the metadata
half of `launch_checked` only -- not the launch, not `cuLaunchKernel`, not the
device time of the copy. Steady-state windows of an `INK_GEN=12` run:

* `=0`, the upload path: **2694--2863 ns**.
* `=1`, a hit: **123--128 ns**, including the two clock reads the timing itself
  costs, so the untimed hit is cheaper still.

About 480 metadata launches a decode step, so ~1.3 ms/step of host-thread time.
The step moved 3.1--3.7 ms, i.e. MORE than the host time removed -- which is the
drop-queue stalls, above, and not the arithmetic of the launches.

Hit rate and working set, same runs: 0 hash collisions in 21,120 metadata
launches; 103 distinct payloads for the WHOLE run (setup + prefill + 12 decode
steps) against a 4096-entry bound, so 0 evictions and the eviction path never
ran outside its unit test. Steady-state decode is 480/480. The first window of
the FIRST pass is already 459/480 (95.6%): the same shape lists recur across the
21 layers within one pass, so the cache pays before any step repeats.

### The prefill: no resolvable effect, and why

Per PREFILL PASS (one forward over the whole 3732-token context), `INK_GEN=1`,
5 interleaved reps per arm, leading arm alternated:

* `=0`: 12919.6 / 12764.0 / 13155.3 / 12415.4 / 12306.1 ms (mean 12712, sd 315)
* `=1`: 12876.1 / 12988.6 / 12035.1 / 12383.8 / 12394.5 ms (mean 12536, sd 351)

Difference 176 ms in the cache's favour, against **+/- 421 ms at 2 sem**. NOT
RESOLVED, and the arithmetic says it never could be: the prefill issues about
5280 metadata launches (setup included), which at ~5 us each is ~28 ms of host
work -- 0.2% of a 12.5 s pass and a fifteenth of this instrument's resolution.
The prefill is device-bound; there is nothing for a host-side saving to move
into. An earlier 3-rep sample showed a 4.7% REGRESSION and that was inside the
same noise. Recorded because the expectation going in was that the prefill --
the part a captured graph lane never reaches -- was where the value would be,
and it is not.

### The capture arm, and smaller captures

`graph_capture_probe 64 5`, `CUBECL_GRAPH_ARENA=0` (the probe predates the
capture arena and never calls `graph_arena_begin`, so every arm fails
identically without it), one GB10:

* nodes in the captured graph: **128** at `=0`, **128** at `=1` (bypassed inside
  a capture, as designed), **64** at `=2`. Every memcpy node gone; the graph is
  exactly its kernels.
* replay e2e per rep: 610.5 +/- 9.5 us at `=0`, **152.4 +/- 0.4 us** at `=2`.
  CAVEAT: this probe's shape is deliberately tiny and L2-resident and its own
  documentation says its device time is not representative of anything. The node
  count is the exact number here; the replay time is an illustration.
* the probe's eager-host figures cannot carry any claim: `=1` and `=2` have
  IDENTICAL eager paths by construction and they measured 581.0 against
  417.2 us/rep, so ~40% is this probe's between-process noise floor.

### Correctness

Bit-identical token stream, same binary, same prompt: 6 runs at `INK_GEN=64`
(3 per arm, interleaved), 2 profiled runs at `INK_GEN=16`, and 2 prefill runs --
every top-5-id-per-position artifact byte-identical across all of them
(`sha256 5d28d320...` for the 64-step set), and the same generated tokens step
for step.

### What is still untested

* `=2` against the real graph lane. It was exercised only on
  `graph_capture_probe`; mary's lane lives on another branch.
* the eviction path on device -- 103 entries against a 4096 bound, so it never
  ran. Unit-tested only.
* several compute streams. This config has one.
* tensor parallelism and the two-node split.
* the four unit tests in `cubecl-cuda/src/compute/meta_cache.rs` could not be
  EXECUTED. `cubecl-cuda` does not build standalone in this workspace: a
  standalone resolution picks crates.io `cubecl-core` 0.10.0, which fails to
  compile at `FastMath::all().difference(FastMath::NotNaN)` before any of this
  code is reached. Every cubecl crate in the two lockfiles is the same version
  AND the same checksum, so the difference is feature unification inside mary's
  larger graph, and it is not reachable by patching or by
  `--no-default-features`. mary's workspace cannot run a dependency's unit tests
  either. The correctness evidence above is the gate, not the tests.
