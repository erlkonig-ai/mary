//! Per-pass system state, for the intermittent multi-second decode stall.
//!
//! # What this exists to decide
//!
//! A decode arm sometimes runs 11x slower than its own twin. Caught cleanly on
//! 2026-08-25 (`/tmp/pipe-e2etopk`, two-node pipe, split 21, ctx 3792, 40
//! passes, 38 warm, `INK_SPEC=1`): arm `t512` rep 1 averaged 1977.1 ms/step and
//! rep 2 of the SAME arm averaged 169.4, with neither carrying `CUBECL_DEBUG_LOG`
//! and both accepting an identical 20/40 drafts. The two reports read together
//! say where it is NOT:
//!
//! | per step, warm     | rep 1 (slow) | rep 2 (fast) |
//! |--------------------|--------------|--------------|
//! | head compute       |   117.3 ms   |   115.4 ms   |
//! | head BLOCKED       |  2244.8 ms   |   116.7 ms   |
//! | tail compute       |  2252.6 ms   |   121.0 ms   |
//! | tail BLOCKED       |   109.6 ms   |   111.2 ms   |
//!
//! The head's own compute is FLAT across an 11x wall difference and the tail's
//! is 18.6x. The whole excess is inside the tail's own pass, and the head is
//! merely blocked on it -- so "both ends stall together" is one stalled node and
//! one waiting one, not a shared external cause. Inside the tail it lands in
//! `mlp short_conv` (9343.4 ms against 111.0 on the twin) and `router + group`'s
//! BLOCKING read (2027.8 against 673.7), which are the two brackets this
//! binary's own `nsys` note names as where DEVICE cost surfaces when nothing
//! else in the loop synchronises. The per-expert ENQUEUE brackets beside them
//! are unmoved -- routed experts 1.7 vs 1.9 ms, shared 3.8 vs 3.9 -- so the host
//! is issuing identical work at an identical rate and waiting longer for it.
//!
//! That is as far as the process's own timers can go. Everything still standing
//! is a property of the machine under the process, and none of it is visible to
//! a timer: what this samples, and which suspect each field kills or convicts:
//!
//! * **`utime`/`stime` against the pass wall.** Host CPU burnt in the pass. A
//!   JIT compile (NVRTC, or the driver's PTX->SASS inside `cuModuleLoadData`)
//!   is host CPU on the launching thread, so a compile stall has CPU ~= wall and
//!   a wait of any kind has CPU << wall. This is the field that makes the
//!   compile hypothesis falsifiable without the 24.5 MB `CUBECL_DEBUG_LOG`,
//!   which costs 3.4x by itself and cannot be carried in a timed run.
//! * **`majflt`, `pgmajfault`, `swap`.** Paging. The 2026-08-25 logs already
//!   report `disk read_bytes 0.00 GiB` and a flat 0.3 GiB of swap across both
//!   twins, so this is expected to stay at zero -- recorded so that "it is not
//!   paging" is evidence rather than an assumption carried forward.
//! * **`anon_thp`, `thp_collapse`, `thp_split`, `compact_stall`.** Transparent
//!   huge pages behind the 77.68 GiB anonymous weight arena. On a unified-memory
//!   part the GPU reads those same pages through the SMMU, so an arena that came
//!   up on 4 KiB pages because memory was fragmented is a TLB problem for the
//!   DEVICE at no cost to the host -- identical work, identical pool, slower
//!   device. It is the one candidate that also predicts the SHAPE of the slow
//!   rep: khugepaged collapses a bounded number of pages per scan, so the pass
//!   time should DECAY over tens of passes rather than switch, and rep 1 decays
//!   5768 -> 3996 -> 3118 -> ... -> 519 ms over 40 passes without ever reaching
//!   the 157 ms floor its twin reached on pass 3. Recorded as a hypothesis with
//!   a matching signature, not as a finding.
//! * **PSI (`/proc/pressure/*`).** How long the kernel stalled this process for
//!   memory, io or cpu. The direct instrument for reclaim, and it distinguishes
//!   "the box was short of memory" from "the box was busy".
//! * **`nvcsw`/`nivcsw`.** Involuntary context switches say another runnable
//!   process arrived; voluntary ones say we blocked. A competing tenant that
//!   shows up mid-rep -- the idle gate samples BEFORE the rep and cannot see
//!   one -- moves `nivcsw`.
//!
//! # Cost, because an instrument carried in a timed run must state one
//!
//! One `getrusage` and five small `/proc` reads per decode pass: `meminfo`,
//! `vmstat`, `self/status`, `self/io` and the three pressure files. Measured on
//! the box it is written for, it is under 200 us a pass against a 157 ms floor,
//! so it is under 0.13% and does not need its own arm. It is still OFF by
//! default (`INK_STEPSTAT=1`), because a run that changes two things at once
//! answers neither question -- the lesson of the `CUBECL_DEBUG_LOG` +
//! autotune-level run that was briefly read as an autotune result.
//!
//! # How to read the output
//!
//! One `[stepstat]` line per decode pass, on its own line and not folded into
//! the existing `step N: ...` line, so every existing parser (`bench-decode.sh`,
//! `pipe-bench.sh`) sees exactly what it saw before. Counters are printed as
//! PER-PASS DELTAS; levels (`avail`, `anon_thp`, PSI averages) are printed as
//! levels. `t` is wall-clock milliseconds since the epoch so that the two nodes'
//! lines and a box-level sampler can be merged on one axis.

#[cfg(target_os = "linux")]
use std::time::{SystemTime, UNIX_EPOCH};

/// Whether the caller should sample at all. `INK_STEPSTAT=1`.
pub fn enabled() -> bool {
    std::env::var("INK_STEPSTAT").ok().as_deref() == Some("1")
}

/// One sample of everything under the process that a timer cannot see.
///
/// Every field is a raw kernel counter or level; nothing is normalised here, so
/// a reader can always recover what was actually read.
#[derive(Default, Clone, Copy)]
pub struct StepStat {
    /// Wall clock, milliseconds since the epoch. The merge key across the two
    /// nodes and any box-level sampler.
    pub t_unix_ms: u128,
    /// Host CPU burnt by the whole process, user and system, in milliseconds.
    /// Compared against the pass wall this is the compile-versus-wait test.
    pub utime_ms: f64,
    pub stime_ms: f64,
    /// Faults and context switches, from `getrusage(RUSAGE_SELF)`.
    pub minflt: u64,
    pub majflt: u64,
    pub nvcsw: u64,
    pub nivcsw: u64,
    /// `/proc/meminfo`, KiB.
    pub avail_kb: u64,
    pub free_kb: u64,
    pub cached_kb: u64,
    pub anon_kb: u64,
    /// `AnonHugePages`. The THP-backed part of the anonymous arena.
    pub anon_thp_kb: u64,
    pub swap_used_kb: u64,
    /// `VmRSS` from `/proc/self/status`, KiB.
    pub rss_kb: u64,
    /// `/proc/vmstat` counters. Cumulative; printed as deltas.
    pub thp_fault_alloc: u64,
    pub thp_collapse_alloc: u64,
    pub thp_split_page: u64,
    pub compact_stall: u64,
    pub pgmajfault: u64,
    pub numa_pages_migrated: u64,
    /// Bytes this process pulled off the block device. A page-cache hit is
    /// free here, which is what makes it a residency instrument.
    pub read_bytes: u64,
    /// `/proc/pressure/*`, the `avg10` of each line, in percent.
    pub psi_cpu_some: f64,
    pub psi_mem_some: f64,
    pub psi_mem_full: f64,
    pub psi_io_some: f64,
    pub psi_io_full: f64,
}

impl StepStat {
    /// Sample now. Cheap enough to call on every decode pass; see the module
    /// note for the measured cost.
    #[cfg(target_os = "linux")]
    pub fn sample() -> Self {
        let mut s = Self {
            t_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
            ..Default::default()
        };

        // SAFETY: `getrusage` writes a fully-initialised `rusage` through the
        // pointer and reads nothing from it. The struct is zeroed first so a
        // failed call leaves defined values rather than garbage.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } == 0 {
            let ms = |t: libc::timeval| t.tv_sec as f64 * 1e3 + t.tv_usec as f64 / 1e3;
            s.utime_ms = ms(ru.ru_utime);
            s.stime_ms = ms(ru.ru_stime);
            s.minflt = ru.ru_minflt as u64;
            s.majflt = ru.ru_majflt as u64;
            s.nvcsw = ru.ru_nvcsw as u64;
            s.nivcsw = ru.ru_nivcsw as u64;
        }

        // `/proc/meminfo`: "Key: <value> kB".
        let mut swapfree = 0u64;
        let mut swaptotal = 0u64;
        if let Ok(txt) = std::fs::read_to_string("/proc/meminfo") {
            for l in txt.lines() {
                let Some((key, rest)) = l.split_once(':') else {
                    continue;
                };
                let v: u64 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                match key {
                    "MemAvailable" => s.avail_kb = v,
                    "MemFree" => s.free_kb = v,
                    "Cached" => s.cached_kb = v,
                    "AnonPages" => s.anon_kb = v,
                    "AnonHugePages" => s.anon_thp_kb = v,
                    "SwapFree" => swapfree = v,
                    "SwapTotal" => swaptotal = v,
                    _ => {}
                }
            }
        }
        s.swap_used_kb = swaptotal.saturating_sub(swapfree);

        // `/proc/vmstat`: "key value". Names that a kernel may not carry are
        // simply absent and stay zero; a field that is structurally zero on
        // this kernel is still worth printing, because "it never moved" and
        // "it does not exist here" both need to be visible rather than
        // inferred.
        if let Ok(txt) = std::fs::read_to_string("/proc/vmstat") {
            for l in txt.lines() {
                let Some((key, val)) = l.split_once(' ') else {
                    continue;
                };
                let v: u64 = val.trim().parse().unwrap_or(0);
                match key {
                    "thp_fault_alloc" => s.thp_fault_alloc = v,
                    "thp_collapse_alloc" => s.thp_collapse_alloc = v,
                    "thp_split_page" => s.thp_split_page = v,
                    "compact_stall" => s.compact_stall = v,
                    "pgmajfault" => s.pgmajfault = v,
                    "numa_pages_migrated" => s.numa_pages_migrated = v,
                    _ => {}
                }
            }
        }

        if let Ok(txt) = std::fs::read_to_string("/proc/self/status") {
            for l in txt.lines() {
                if let Some(rest) = l.strip_prefix("VmRSS:") {
                    s.rss_kb = rest
                        .split_whitespace()
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                }
            }
        }

        if let Ok(txt) = std::fs::read_to_string("/proc/self/io") {
            for l in txt.lines() {
                if let Some(rest) = l.strip_prefix("read_bytes:") {
                    s.read_bytes = rest.trim().parse().unwrap_or(0);
                }
            }
        }

        // PSI. Absent entirely on a kernel built without it, which is itself
        // worth knowing: the fields stay at zero and the reader can tell,
        // because a box under real pressure never reports a clean 0.00 on
        // every one of five lines for a whole run.
        let psi = |path: &str, want: &str| -> f64 {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|txt| {
                    txt.lines()
                        .find(|l| l.starts_with(want))
                        .and_then(|l| {
                            l.split_whitespace()
                                .find_map(|f| f.strip_prefix("avg10="))
                                .map(str::to_string)
                        })
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(0.0)
        };
        s.psi_cpu_some = psi("/proc/pressure/cpu", "some");
        s.psi_mem_some = psi("/proc/pressure/memory", "some");
        s.psi_mem_full = psi("/proc/pressure/memory", "full");
        s.psi_io_some = psi("/proc/pressure/io", "some");
        s.psi_io_full = psi("/proc/pressure/io", "full");
        s
    }

    /// Everything above is a `/proc` reading, so off Linux there is nothing to
    /// sample and the caller still compiles.
    #[cfg(not(target_os = "linux"))]
    pub fn sample() -> Self {
        Self::default()
    }

    /// One line, this sample against the previous one.
    ///
    /// Counters print as PER-PASS DELTAS and levels print as levels, because a
    /// cumulative fault count over a 40-pass run answers no question anybody
    /// asked and a delta on `MemAvailable` hides how close to the floor it was.
    ///
    /// `cpu` is the fraction of the pass the process spent on a CPU: at 1.0 the
    /// host was computing for the whole pass (a compile, or host-side work), at
    /// ~0.0 it was waiting. That ratio is the single most load-bearing field
    /// here and it is why the line carries `pass_ms` as well.
    pub fn line(&self, prev: &Self, step: usize, pass_ms: f64) -> String {
        let d = |now: u64, was: u64| now.saturating_sub(was);
        let cpu_ms = (self.utime_ms - prev.utime_ms) + (self.stime_ms - prev.stime_ms);
        let g = |kb: u64| kb as f64 / (1u64 << 20) as f64;
        format!(
            "  [stepstat] step {step} t {} pass_ms {:.1} cpu_ms {:.1} cpu_frac {:.3} \
             (u {:.1} s {:.1}) minflt {} majflt {} nvcsw {} nivcsw {} \
             avail {:.2} free {:.2} cached {:.2} anon {:.2} anon_thp {:.2} swap {:.2} rss {:.2} \
             thp_fault {} thp_collapse {} thp_split {} compact_stall {} pgmajfault {} \
             numa_migrated {} disk_read_mib {:.1} \
             psi_cpu {:.2} psi_mem {:.2}/{:.2} psi_io {:.2}/{:.2}",
            self.t_unix_ms,
            pass_ms,
            cpu_ms,
            if pass_ms > 0.0 { cpu_ms / pass_ms } else { 0.0 },
            self.utime_ms - prev.utime_ms,
            self.stime_ms - prev.stime_ms,
            d(self.minflt, prev.minflt),
            d(self.majflt, prev.majflt),
            d(self.nvcsw, prev.nvcsw),
            d(self.nivcsw, prev.nivcsw),
            g(self.avail_kb),
            g(self.free_kb),
            g(self.cached_kb),
            g(self.anon_kb),
            g(self.anon_thp_kb),
            g(self.swap_used_kb),
            g(self.rss_kb),
            d(self.thp_fault_alloc, prev.thp_fault_alloc),
            d(self.thp_collapse_alloc, prev.thp_collapse_alloc),
            d(self.thp_split_page, prev.thp_split_page),
            d(self.compact_stall, prev.compact_stall),
            d(self.pgmajfault, prev.pgmajfault),
            d(self.numa_pages_migrated, prev.numa_pages_migrated),
            d(self.read_bytes, prev.read_bytes) as f64 / (1u64 << 20) as f64,
            self.psi_cpu_some,
            self.psi_mem_some,
            self.psi_mem_full,
            self.psi_io_some,
            self.psi_io_full,
        )
    }
}
