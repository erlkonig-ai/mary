//! Which memory pool a prefill wants, and when to hand its pages back.
//!
//! [`super::budget`] is about the per-BUFFER cap — one comparison against a
//! number the device reports, made before the run spends a minute copying
//! weights. This module is about the other thing the allocator does to a
//! prefill, which is larger and was not being looked at: it RESERVES three to
//! five times what the prefill holds, and on a unified-memory part the reserved
//! figure is what the node runs out of.
//!
//! # The ladder, and the holes in it
//!
//! cubecl's default configuration is `MemoryConfiguration::SubSlices`. It
//! builds pools from `max_page_size = cuDeviceTotalMem / 4` downwards, dividing
//! by four each rung — 30.4, 7.6, 1.9, 0.475 GiB on a 121.6 GiB part — and each
//! rung takes slices up to `page_size / 2^k` for its own `k`, not up to
//! `page_size`. That second parameter is the one that is easy to miss and it is
//! the one that hurts. Written out for this part:
//!
//! | page | largest slice it takes | so it serves |
//! |---|---|---|
//! | 0.475 GiB | 0.059 GiB | up to 0.059, or 0.38 .. 0.475 |
//! | 1.9 GiB | 0.475 GiB | up to 0.475, or 1.52 .. 1.9 |
//! | 7.6 GiB | 3.8 GiB | up to 3.8, or 6.08 .. 7.6 |
//! | 30.4 GiB | 30.4 GiB | everything else |
//!
//! (The "or" is a second rule: a pool also accepts a request within 20% of its
//! whole page.) Read the gaps rather than the rows. A buffer of 0.6 GiB is too
//! big for the 1.9 GiB page's slice cap and too small for its 20% rule, so it
//! goes to the 7.6. A buffer of 4 GiB is too big for the 7.6's slice cap and
//! too small for its 20% rule, so it goes to **the 30.4**. The routed lane's
//! gather is 4.12 GiB at f32 and 2.06 at BF16 at 42,000 tokens, and those two
//! land in different pools by a factor of four in page size while differing by
//! a factor of two in bytes.
//!
//! This is why the previous study's conclusion — HALVING A BUFFER OFTEN
//! RESERVES THE SAME — was right and also not the whole shape of it. Halving
//! can reserve the same, and it can reserve a quarter, and which one happens is
//! decided by two thresholds that have nothing to do with the model.
//!
//! # What to do about it, which is not to size buffers to page boundaries
//!
//! The obvious repair is to round every large allocation UP to whichever page
//! it is going to occupy anyway, so that a page holds one buffer exactly and
//! nothing is stranded. It is the wrong repair: the thresholds move with
//! `cuDeviceTotalMem`, so the arithmetic would be device-specific and silently
//! wrong on the next part; and it makes every buffer larger in `live` to make
//! it smaller in `reserved`, which on unified memory is the same memory.
//!
//! The right repair is to stop using a fixed ladder that this workload does not
//! fit. `MemoryConfiguration::ExclusivePages` has no ladder: twenty-four
//! log-spaced buckets, and a page is allocated at the size of the REQUEST
//! (rounded to the average the bucket has seen, then to alignment). It is
//! reachable already, on this fork, as `CUBECL_MEMORY_CONFIG=exclusive`, and
//! [`choose_memory_config`] selects it when the operator has not.
//!
//! Measured, one node, eight layers, 42,000 tokens, `INK_QBLOCK=64`, six arms
//! of one binary. `INK_MEM_TRACE=1` throughout, so the elapsed column is a
//! traced pass and comparable across rows but not to an untraced one.
//!
//! | arm | peak pool reserved | live | peak node used | elapsed |
//! |---|---|---|---|---|
//! | subslices, no cleanup (the default until now) | 50.63 GiB | 9.71 | 97.37 | 46.2 s |
//! | subslices + per-layer cleanup | 40.53 | 9.71 | 79.32 | 48.0 s |
//! | exclusive, no cleanup | 30.12 | 9.71 | 76.24 | 45.7 s |
//! | exclusive + per-layer cleanup | 15.63 | 9.71 | 61.29 | 47.7 s |
//! | + BF16 residual stream | 15.32 | 9.06 | 57.24 | 47.7 s |
//! | + per-STAGE cleanup | **9.21** | 9.06 | 56.00 | 50.6 s |
//!
//! Read the `reserved` column, not `peak node used`. Reserved is exact and
//! reproduced to the hundredth of a gibibyte across two separate binaries;
//! peak-used carries several gibibytes of page cache and is only good to about
//! that. The two agree on every ordering and on roughly every delta, which is
//! the reason to trust either.
//!
//! The last row is the one worth staring at: reserved 9.21 against live 9.06.
//! Cleaning up at every stage boundary rather than at the end of the layer
//! leaves the pool holding 150 MiB it is not using, on a workload where the
//! default allocator was holding forty gibibytes. It is not the default anyway,
//! because it costs five syncs a layer instead of one and the memory it saves
//! over the per-layer policy is 1.2 GiB of node -- which buys nothing that a
//! thousand more tokens would not, and costs time on every run.
//!
//! # `max_page_size` is not the knob
//!
//! It cannot be set: `CudaRuntime`'s `init` writes `cuDeviceTotalMem / 4` into
//! `MemoryDeviceProperties` once and nothing reads an override. It could be
//! given one, and the tempting sum is that a 121.6 GiB part with 82.73 GiB of
//! weights resident has about 35 GiB to play with, so a `max_page_size` of 8
//! GiB would put the ladder's rungs at 8 / 2 / 0.5 and fit the prefill's
//! buffers better.
//!
//! It would not be worth having. It narrows the per-buffer cap that
//! [`super::budget::check`] enforces — 30.41 GiB is what makes a two-million-
//! token sequence admissible on paper — in exchange for rungs that are still a
//! fixed ladder with the same two thresholds and the same holes, just at
//! different sizes. `ExclusivePages` removes the ladder instead of retuning it,
//! costs nothing, and needs no number chosen per device.
//!
//! # When to hand pages back
//!
//! Every layer's activations are freed before the next layer allocates its own,
//! so between two layers the pool holds almost nothing and reserves almost
//! everything. `memory_cleanup` fixes that, and the reason it was a switch and
//! not a default is real: a page handed back is a `cudaFree` the next layer
//! pays a `cudaMalloc` for.
//!
//! That cost is now measured rather than argued. Warm passes, one box, eight
//! layers, 20,000 tokens, five passes with the first discarded, p50:
//!
//! | arm | p50 pass |
//! |---|---|
//! | exclusive, no cleanup | 9,565 ms |
//! | exclusive + per-layer cleanup | 10,428 ms |
//! | exclusive + per-STAGE cleanup | 10,944 ms |
//!
//! Nine percent for the per-layer policy, another eight for the per-stage one.
//! And at 20,000 tokens on a box with seventy gibibytes spare it buys NOTHING:
//! the run was never going to fail. At 130,000 tokens on the two-node split it
//! is the difference between a prefill and an OOM kill.
//!
//! [`CleanupPolicy::WhenStranded`] on the same probes, and the count of layers
//! that actually cleaned up is printed by the run rather than inferred:
//!
//! | probe | policy | cleanups | result |
//! |---|---|---|---|
//! | 20,000 tokens, 8 layers | always | 8 of 8 | p50 pass 10,137 ms |
//! | 20,000 tokens, 8 layers | **when stranded** | **0 of 8** | p50 pass 9,474 ms |
//! | 42,000 tokens, 8 layers | always | 8 of 8 | 15.32 GiB reserved, 57.46 node |
//! | 42,000 tokens, 8 layers | **when stranded** | **0 of 8** | 25.62 GiB reserved, 71.49 node |
//!
//! Both of those single-box probes leave seventy gibibytes of the node spare,
//! so the policy declines to pay and the run is six and a half percent faster
//! for it. The 42,000-token row is the one to read carefully: it reserves ten
//! gibibytes MORE than the always-clean arm and that is the correct answer,
//! because those ten gibibytes were free and nothing wanted them.
//!
//! So neither "always" nor "never" is right, and neither is "is this a prefill"
//! -- a small prefill wants it as little as a decode step does. What decides it
//! is whether the pool is holding more than the node can spare, which is two
//! numbers this process can read: `MemoryUsage::bytes_reserved` minus
//! `bytes_in_use` on one side, and `MemAvailable` (cgroup limit included) on the
//! other.
//!
//! [`CleanupPolicy::WhenStranded`] is that comparison and it is the default. No
//! threshold is chosen: the condition is the thing itself. A pool stranding
//! more than the node has free is a pool whose next allocation may be the one
//! that fails, and a pool stranding less than that is a pool nobody has to pay
//! nine percent for. A decode step reaches it too and takes the cheap branch
//! for the same reason, without anyone having to say "decode" anywhere.
//!
//! `INK_POOL_CLEANUP=0|1|stage` still overrides it in either direction for the
//! A/B, and the header prints which policy ran.
//!
//! Confirmed where it matters rather than inferred: at 130,000 and at 140,000
//! tokens on the two-node split the run reports `pool cleanups: 20 of 20` on the
//! head and `22 of 22` on the tail. The policy takes the expensive branch at the
//! sizes that need it and the cheap one at the sizes that do not, and the count
//! is in the log either way.

/// The allocator strategy selected before cubecl creates a CUDA context.
///
/// This is kept as an application type rather than exposing cubecl's enum so
/// admission can price the exact same choice without constructing a runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocatorConfig {
    /// Fixed page-size ladder; large buffers strand the two top rungs.
    SubSlices,
    /// One request-sized page per allocation. Retention is lower in the
    /// measured runs, but remains dependent on the request history.
    ExclusivePages,
}

impl AllocatorConfig {
    fn parse(value: Option<&str>) -> Self {
        match value {
            Some("subslices") => Self::SubSlices,
            Some("exclusive") | None => Self::ExclusivePages,
            Some(other) => panic!(
                "CUBECL_MEMORY_CONFIG={other:?} is not recognised \
                 (expected \"exclusive\" or \"subslices\")"
            ),
        }
    }

    /// The value cubecl reads from `CUBECL_MEMORY_CONFIG`.
    pub fn env_value(self) -> &'static str {
        match self {
            Self::SubSlices => "subslices",
            Self::ExclusivePages => "exclusive",
        }
    }

    /// Allocator reservation charged before the startup copy.
    ///
    /// SubSlices allocates the two largest ladder pages for this workload.
    /// ExclusivePages removes that fixed ladder, but it does not remove
    /// retention: its 24 first-fit buckets keep one request-sized page per
    /// allocation, round new pages to a per-bucket moving average, and reclaim
    /// them only after their handles are free. The aggregate admission model
    /// does not carry the ordered allocation trace needed to prove a smaller
    /// bound, so it retains the existing conservative charge. This is
    /// intentionally stricter than the measured ExclusivePages high-water
    /// marks (30.12 GiB without explicit cleanup and 15.63 GiB with per-layer
    /// cleanup on a 121.6 GiB node), rather than turning those one-workload
    /// observations into an unsafe universal coefficient.
    /// The prefill length the un-scaled floor was sized for.
    ///
    /// `machine / 4 + machine / 16` is not a fitted coefficient -- it is two
    /// page sizes off a ladder the runtime derives from the device -- but it
    /// was measured against a 16,384-token PREFILL. Charging it to a decode
    /// step is charging a constant for something linear in the sequence, which
    /// is the exact error `Checkpoint::copy_share` documents itself against:
    /// "folding one into the other is how the gate came to charge a constant
    /// for something linear in the sequence".
    ///
    /// Measured 2026-08-25, spark-zt, INK_LAYERS=0:16, 256-token prompt,
    /// decode, cleanup OFF (the worst arm): admission charged 45.20 GiB of
    /// context/activations while the pool reserved 1.88 GiB -- 1.42 live, 0.46
    /// stranded. The floor was ~20x the high-water, and that is what refuses
    /// ranges that would run.
    const PREFILL_REFERENCE: u64 = 16_384;

    pub fn admission_floor(self, machine: u64, prefill_tokens: usize) -> u64 {
        match self {
            Self::SubSlices | Self::ExclusivePages => {
                let full = machine / 4 + machine / 16;
                let scaled = full.saturating_mul(prefill_tokens as u64) / Self::PREFILL_REFERENCE;
                scaled.clamp(machine / 16, full)
            }
        }
    }
}

/// Select the cubecl memory configuration, unless the operator already has.
///
/// Called before anything can create a CUDA context, because
/// `CUBECL_MEMORY_CONFIG` is read once in `CudaRuntime`'s `init` and a value
/// set after the first client exists is a value that does nothing.
///
/// Setting an environment variable from inside the process is a blunt
/// instrument and is used here on purpose: the fork exposes this choice at
/// exactly one place, and the alternative is to ask every operator of every
/// harness to remember a variable whose right value is a property of the
/// workload rather than of the machine. It is still an override -- a value
/// already present is left alone, including a deliberate `subslices`.
pub fn choose_memory_config() -> AllocatorConfig {
    const VAR: &str = "CUBECL_MEMORY_CONFIG";
    let override_value = std::env::var(VAR).ok();
    let config = AllocatorConfig::parse(override_value.as_deref());
    if override_value.is_some() {
        return config;
    }
    // SAFETY: single-threaded, first statement of `main` after `fatal::arm`,
    // before any client, thread or context exists. The `unsafe` is edition
    // 2024's, not this edition's, and is written anyway so that the safety
    // argument is attached to the call rather than lost in a future bump.
    #[allow(unused_unsafe)]
    unsafe {
        std::env::set_var(VAR, config.env_value())
    };
    config
}

/// Whether this pass hands the pool's unused pages back, and how often.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CleanupPolicy {
    /// Never. `INK_POOL_CLEANUP=0`.
    Never,
    /// Between layers, when the pool is stranding more than the node has spare.
    /// The default; see the module doc for why the comparison is the whole
    /// policy and there is no threshold in it.
    WhenStranded,
    /// Between layers, unconditionally. `INK_POOL_CLEANUP=1`.
    PerLayer,
    /// Between stages within a layer -- five syncs a layer instead of one.
    /// `INK_POOL_CLEANUP=stage`.
    PerStage,
}

impl CleanupPolicy {
    /// The policy for this run, from `INK_POOL_CLEANUP`.
    pub fn choose() -> CleanupPolicy {
        match std::env::var("INK_POOL_CLEANUP").as_deref() {
            Ok("0") => CleanupPolicy::Never,
            Ok("1") => CleanupPolicy::PerLayer,
            Ok("stage") => CleanupPolicy::PerStage,
            Ok("auto") | Err(_) => CleanupPolicy::WhenStranded,
            Ok(other) => panic!(
                "INK_POOL_CLEANUP={other:?} is not recognised \
                 (expected \"0\", \"1\", \"stage\" or \"auto\")"
            ),
        }
    }

    /// Whether to clean up at the end of a layer, given the pool's stranded
    /// bytes, when the policy is [`CleanupPolicy::WhenStranded`].
    ///
    /// For [`CleanupPolicy::PerStage`] this is the tail-stage hand-back. The
    /// caller defers it until after the layer's RMS diagnostic so that the
    /// diagnostic's full-width temporary is released too; the four internal
    /// routed-stage hand-backs happen at their own boundaries.
    ///
    /// A node whose `MemAvailable` cannot be read gets the cheap branch: an
    /// unreadable `/proc/meminfo` is not a reason to start paying nine percent.
    ///
    /// Prefer [`CleanupGate`] at a call site inside the layer loop: this
    /// function's `stranded` argument is not free to produce, and the two
    /// policies that ignore it should not be paying for it.
    pub fn at_layer(self, stranded: u64) -> bool {
        match self {
            CleanupPolicy::Never => false,
            CleanupPolicy::PerLayer | CleanupPolicy::PerStage => true,
            CleanupPolicy::WhenStranded => match super::pile::mem_available_bytes() {
                Ok(avail) => stranded > avail,
                Err(_) => false,
            },
        }
    }

    /// Whether to clean up at each stage boundary inside a layer.
    pub fn at_stage(self) -> bool {
        self == CleanupPolicy::PerStage
    }

    /// What to print in the run's header.
    pub fn name(self) -> &'static str {
        match self {
            CleanupPolicy::Never => "off",
            CleanupPolicy::WhenStranded => {
                "between layers, when the pool strands more than the node has spare"
            }
            CleanupPolicy::PerLayer => "between layers, always",
            CleanupPolicy::PerStage => "between stages",
        }
    }
}

/// How often the policy's question is ASKED, which is not the same question as
/// what it answers.
///
/// [`CleanupPolicy::at_layer`] takes `stranded` as an argument and says nothing
/// about where it comes from. It comes from `ComputeClient::memory_usage`, and
/// that call is not the free host-side bookkeeping the comment on it used to
/// claim. On this cubecl lineage the compute server lives on its own thread and
/// `memory_usage` is `submit_blocking`: it enqueues a closure behind everything
/// the layer has already submitted, wakes the runner, and blocks the caller on a
/// oneshot until the runner has drained down to it. So the call is a HOST
/// LAUNCH-QUEUE BARRIER, and once it arrives at the runner it walks every page
/// of all twenty-four `ExclusivePages` buckets and every slice of the persistent
/// pool, allocating a `Vec` of the live ones per pool as it goes.
///
/// That was being paid once per layer, in every arm, under every policy --
/// including [`CleanupPolicy::Never`] and [`CleanupPolicy::PerLayer`], neither
/// of which reads `stranded` at all. It is the reason an `INK_POOL_CLEANUP=0`
/// A/B measured +0.00%: the switch it flips is downstream of the cost.
///
/// The gate fixes both halves without moving the policy:
///
/// * A policy that does not read `stranded` never asks for it. `Never` and
///   `PerLayer`/`PerStage` are decided from the policy alone.
/// * [`CleanupPolicy::WhenStranded`] asks per LAYER while the answer is yes and
///   once per PASS while it is no. The two regimes the policy was confirmed in
///   are both constant across a pass -- `0 of 42` on a decode step, `20 of 20`
///   and `22 of 22` on the two-node prefill at 130,000 tokens -- so the sample
///   that is dropped is a sample whose answer the neighbouring samples already
///   carried. The pass's LAST layer always polls, so a pass that has gone quiet
///   still re-arms per-layer polling for the next one, and the first answer of
///   `yes` re-arms it for the rest of the current one.
///
/// What that trades is exactly one pass of latency on a transition from "not
/// stranding" to "stranding", and the bound on it is the pass's own
/// allocations: a decode step whose per-layer activations are megabytes cannot
/// cross a gibibyte-scale margin inside one pass, and a prefill that can is a
/// pass the previous poll already answered yes for.
///
/// `INK_POOL_POLL=layer` restores the unconditional per-layer poll, in every
/// policy, so the cost of the poll can be A/B'd against its absence inside ONE
/// binary and one worktree rather than across two.
#[derive(Clone, Copy, Debug)]
pub struct CleanupGate {
    policy: CleanupPolicy,
    /// `INK_POOL_POLL=layer`: poll unconditionally, as the loop used to.
    always_poll: bool,
    /// Poll at every layer of the pass now running.
    per_layer: bool,
    /// Some poll in the pass now running answered yes.
    acted: bool,
}

impl CleanupGate {
    /// The gate for this run. Starts armed, so the first pass polls per layer
    /// and nothing has to be assumed about what state the pool woke up in.
    pub fn new(policy: CleanupPolicy) -> Self {
        Self {
            policy,
            always_poll: matches!(std::env::var("INK_POOL_POLL").as_deref(), Ok("layer")),
            per_layer: true,
            acted: false,
        }
    }

    /// What the run's header should say about the sampling schedule.
    pub fn schedule(&self) -> &'static str {
        if self.always_poll {
            "every layer (INK_POOL_POLL=layer)"
        } else {
            match self.policy {
                CleanupPolicy::Never | CleanupPolicy::PerLayer | CleanupPolicy::PerStage => {
                    "never -- this policy does not read the pool"
                }
                CleanupPolicy::WhenStranded => {
                    "every layer while it is handing pages back, once a pass while it is not"
                }
            }
        }
    }

    /// Call at the top of every pass, before its first layer.
    pub fn begin_pass(&mut self) {
        self.per_layer = self.acted;
        self.acted = false;
    }

    /// Whether this layer hands the pool's pages back.
    ///
    /// `last` marks the pass's final layer, which always polls. `stranded` is a
    /// closure and not a value because calling it is the cost this type exists
    /// to stop paying: it is invoked only on a layer whose answer can depend on
    /// it.
    pub fn at_layer(&mut self, last: bool, stranded: impl FnOnce() -> u64) -> bool {
        if self.always_poll {
            // The old shape, kept callable for the A/B: read the pool first and
            // unconditionally, then ask the policy.
            let s = stranded();
            let yes = self.policy.at_layer(s);
            if yes {
                self.acted = true;
            }
            return yes;
        }
        match self.policy {
            CleanupPolicy::Never => false,
            CleanupPolicy::PerLayer | CleanupPolicy::PerStage => {
                self.acted = true;
                true
            }
            CleanupPolicy::WhenStranded => {
                if !self.per_layer && !last {
                    return false;
                }
                let avail = match super::pile::mem_available_bytes() {
                    Ok(avail) => avail,
                    Err(_) => return false,
                };
                let yes = stranded() > avail;
                if yes {
                    self.acted = true;
                    self.per_layer = true;
                }
                yes
            }
        }
    }
}

/// What the pool is holding and not using, from a [`MemoryUsage`]-shaped pair.
///
/// `bytes_reserved - (bytes_in_use + bytes_padding)`, saturating: the same
/// quantity `seam::pool_line` prints as STRANDED, named once so the policy and
/// the report cannot drift apart.
pub fn stranded_bytes(reserved: u64, in_use: u64, padding: u64) -> u64 {
    reserved.saturating_sub(in_use + padding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_floor_stays_conservative_for_both_strategies() {
        let machine = 128 << 30;
        assert_eq!(
            AllocatorConfig::SubSlices.admission_floor(machine, 16_384),
            40 << 30
        );
        assert_eq!(
            AllocatorConfig::ExclusivePages.admission_floor(machine, 16_384),
            40 << 30
        );
    }

    #[test]
    fn exclusive_floor_dominates_the_recorded_within_layer_peak() {
        const GIB: u64 = 1 << 30;
        // The probe documented above: 121.6 GiB machine, 9.71 GiB live and
        // 30.12 GiB reserved before an explicit cleanup boundary. Admission
        // adds its allocator floor to its independently modelled live set, so
        // compare the floor with the adversarial retained portion.
        let machine = 1216 * GIB / 10;
        let live = 971 * GIB / 100;
        let reserved = 3012 * GIB / 100;
        let retained = reserved - live;
        let floor = AllocatorConfig::ExclusivePages.admission_floor(machine, 16_384);
        assert!(floor >= retained);
        assert!(live + floor >= reserved);
    }

    #[test]
    fn unset_allocator_selects_exclusive_pages() {
        assert_eq!(
            AllocatorConfig::parse(None),
            AllocatorConfig::ExclusivePages
        );
        assert_eq!(
            AllocatorConfig::parse(Some("subslices")),
            AllocatorConfig::SubSlices
        );
        assert_eq!(
            AllocatorConfig::parse(Some("exclusive")),
            AllocatorConfig::ExclusivePages
        );
    }
}
