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
