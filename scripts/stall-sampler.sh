#!/usr/bin/env bash
#
# stall-sampler.sh -- the box under the run, sampled on a clock, for the
# intermittent multi-second decode stall.
#
# WHY THIS EXISTS SEPARATELY FROM `INK_STEPSTAT`
#
# The two-node evidence of 2026-08-25 (`/tmp/pipe-e2etopk`, split 21, ctx 3792,
# 40 passes, 38 warm, INK_SPEC=1) puts the whole of an 11x twin-rep difference
# inside the TAIL's own pass -- tail compute 2252.6 ms/step against 121.0 on the
# twin, while the head's own compute is 117.3 against 115.4 and the head is
# simply blocked. Inside the tail it surfaces in `mlp short_conv` and `router +
# group`'s BLOCKING read, which this binary's own nsys note names as the two
# brackets that absorb DEVICE time when nothing else in the loop synchronises,
# while the per-expert ENQUEUE brackets beside them do not move. So the host is
# issuing identical work and waiting longer for the device to finish it.
#
# `INK_STEPSTAT=1` samples what the PROCESS can see. Three of the remaining
# suspects are invisible from inside it and this samples those:
#
#   * SM/memory CLOCKS and the driver's throttle reasons. A power or thermal cap
#     is a device slowdown with no host signature at all.
#   * ANOTHER TENANT on the GPU. Both benchmark harnesses gate on an idle box
#     BEFORE the rep and cannot see a process that arrives during it -- and on a
#     box several agents share, arriving during it is the normal case. (The
#     e2etopk run's own gate said "compute-apps: none" before rep 1 and "!! THE
#     BOXES DID NOT STAY IDLE" after the run, which is exactly the resolution
#     that cannot answer this.)
#   * The CPU governor's current frequency, for the same reason on the host side.
#
# It also re-reads the memory, huge-page and pressure counters that `INK_STEPSTAT`
# reads, deliberately: sampled on a fixed clock they show the SHAPE between two
# passes, and they let a run with no instrumented binary still be diagnosed.
#
# COST, because a sampler that runs beside a measurement must state one
#
#   --proc-only : /proc reads only. No GPU interaction whatsoever, a few hundred
#                 microseconds a tick, safe to run beside somebody else's timed
#                 run without asking. This is the default.
#   (full)      : adds ONE long-lived `nvidia-smi ... -l <interval>` stream --
#                 one process for the whole session, not one per tick -- plus a
#                 compute-apps query every APPS_EVERY ticks. Both are driver
#                 queries; neither submits work. Still: it is GPU interaction,
#                 so it is opt-in.
#
# USAGE
#
#   scripts/stall-sampler.sh [--gpu] [--interval SEC] [--out PATH] [--label TXT]
#
#     --gpu           add the nvidia-smi stream (default: /proc only)
#     --interval SEC  sample period, default 0.5
#     --out PATH      TSV destination, default /tmp/stall-sampler-<host>-<ts>.tsv
#     --label TXT     free text recorded in the header, e.g. the arm being run
#     --duration SEC  stop after this long (default: until killed)
#
# READING IT
#
# `ts_ms` is wall-clock milliseconds since the epoch, the same axis the
# `[stepstat]` lines print, so the two merge with a join and no clock alignment.
# Counter columns are CUMULATIVE as the kernel reports them; take deltas when
# reading. Level columns are levels. Nothing is averaged here, because a sampler
# that smooths cannot show the thing it was pointed at.
set -uo pipefail

INTERVAL=0.5
GPU=0
OUT=""
LABEL=""
DURATION=0
APPS_EVERY=4

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=1; shift;;
    --proc-only) GPU=0; shift;;
    --interval) INTERVAL=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --label) LABEL=$2; shift 2;;
    --duration) DURATION=$2; shift 2;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//;$d'; exit 0;;
    *) printf '!! unknown option %s (try --help)\n' "$1" >&2; exit 2;;
  esac
done

HOST=$(hostname)
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-/tmp/stall-sampler-$HOST-$TS.tsv}
GPUSTREAM=""
GPUPID=""

cleanup() {
  [ -n "$GPUPID" ] && kill "$GPUPID" 2>/dev/null
  [ -n "$GPUSTREAM" ] && rm -f "$GPUSTREAM"
}
trap cleanup EXIT INT TERM

# --- the GPU stream, if asked for -------------------------------------------
#
# ONE process for the session. Spawning nvidia-smi per tick costs 40-90 ms of
# driver query each time and would be a perturbation rather than an observation.
if [ "$GPU" = 1 ]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "!! --gpu asked for but no nvidia-smi" >&2; exit 2; }
  GPUSTREAM=$(mktemp /tmp/stall-sampler-gpu.XXXXXX)
  nvidia-smi \
    --query-gpu=utilization.gpu,utilization.memory,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used \
    --format=csv,noheader,nounits -l 1 > "$GPUSTREAM" 2>/dev/null &
  GPUPID=$!
fi

{
  echo "# stall-sampler on $HOST at $TS"
  echo "# label: ${LABEL:-none}"
  echo "# interval: ${INTERVAL}s   gpu stream: $([ "$GPU" = 1 ] && echo yes || echo 'no (proc-only)')"
  echo "# kernel: $(uname -r)"
  echo "# thp enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
  echo "# thp defrag:  $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo n/a)"
  echo "# khugepaged pages_to_scan: $(cat /sys/kernel/mm/transparent_hugepage/khugepaged/pages_to_scan 2>/dev/null || echo n/a)"
  echo "# khugepaged scan_sleep_ms: $(cat /sys/kernel/mm/transparent_hugepage/khugepaged/scan_sleep_millisecs 2>/dev/null || echo n/a)"
  printf 'ts_ms\tgpu_util\tgpu_memutil\tsm_clk\tmem_clk\ttemp\tpower\tthrottle\tgpu_mem_used\tgpu_apps\t'
  printf 'mem_avail_kb\tmem_free_kb\tanon_kb\tanon_thp_kb\tswap_used_kb\t'
  printf 'thp_fault_alloc\tthp_collapse_alloc\tthp_split_page\tcompact_stall\tpgmajfault\tnuma_migrated\t'
  printf 'psi_cpu_some\tpsi_mem_some\tpsi_mem_full\tpsi_io_some\tpsi_io_full\t'
  printf 'load1\tprocs_running\tcpu_khz\tink_pids\n'
} > "$OUT"

echo "stall-sampler: writing $OUT (interval ${INTERVAL}s, $([ "$GPU" = 1 ] && echo 'gpu+proc' || echo 'proc-only')). Ctrl-C to stop."

START=$(date +%s)
TICK=0
APPS="-"
INKPIDS="-"
declare -A MI VS

# Everything in the tick below is a builtin or a redirect, with no forks at all
# on a proc-only tick: `EPOCHREALTIME` instead of `date`, `read` instead of
# `cat`, a parse loop instead of `grep`+`sed`. That is what makes the "a few
# hundred microseconds" in the header true rather than aspirational -- an
# earlier draft forked fifteen processes a tick for the pressure files alone,
# which is a perturbation of the thing being watched.
while :; do
  TICK=$((TICK + 1))
  # EPOCHREALTIME is "<seconds>.<microseconds>"; the locale can make that
  # separator a comma, so split on either.
  ER=${EPOCHREALTIME}
  TS_MS=$(( ${ER%%[.,]*} * 1000 + 10#${ER##*[.,]} / 1000 ))

  # -- GPU ------------------------------------------------------------------
  GU="-"; GMU="-"; SMC="-"; MEMC="-"; TEMP="-"; PWR="-"; THR="-"; GMEM="-"
  if [ "$GPU" = 1 ] && [ -s "$GPUSTREAM" ]; then
    IFS=',' read -r GU GMU SMC MEMC TEMP PWR THR GMEM <<< "$(tail -n 1 "$GPUSTREAM")"
    # strip the spaces nvidia-smi puts after each comma
    GU=${GU// /}; GMU=${GMU// /}; SMC=${SMC// /}; MEMC=${MEMC// /}
    TEMP=${TEMP// /}; PWR=${PWR// /}; THR=${THR// /}; GMEM=${GMEM// /}
    if [ $((TICK % APPS_EVERY)) -eq 0 ]; then
      APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
             | tr '\n' ';' | tr -d ' ')
      APPS=${APPS:--}
    fi
  fi

  # -- /proc/meminfo and /proc/vmstat, one read each -------------------------
  # `read` into an associative array beats a grep per key: this loop runs twice
  # a second for the length of somebody else's benchmark.
  while read -r k v _; do MI[${k%:}]=$v; done < /proc/meminfo
  while read -r k v; do VS[$k]=$v; done < /proc/vmstat

  # -- pressure, parsed in the shell ----------------------------------------
  # Sets PSI_SOME / PSI_FULL from one file. A kernel without PSI leaves both at
  # "-", which is itself a reading: a box under real pressure never reports a
  # clean nothing on every line for a whole run, so "-" and "0.00" must not be
  # allowed to look alike.
  psi_read() {
    PSI_SOME="-"; PSI_FULL="-"
    local kind rest f
    while read -r kind rest; do
      for f in $rest; do
        case "$f" in
          avg10=*) [ "$kind" = some ] && PSI_SOME=${f#avg10=} || PSI_FULL=${f#avg10=};;
        esac
      done
    done < "$1" 2>/dev/null
  }
  psi_read /proc/pressure/cpu;    PSI_CPU=$PSI_SOME
  psi_read /proc/pressure/memory; PSI_MEM_S=$PSI_SOME; PSI_MEM_F=$PSI_FULL
  psi_read /proc/pressure/io;     PSI_IO_S=$PSI_SOME;  PSI_IO_F=$PSI_FULL

  read -r L1 _ _ RUN _ < /proc/loadavg
  PR=${RUN%%/*}

  CPUKHZ="-"
  read -r CPUKHZ < /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || CPUKHZ="-"

  # The only fork on a proc-only tick, and it is throttled to one in
  # APPS_EVERY: how many of our own decode processes are on this box. A second
  # one arriving is the cheapest possible explanation for a stall and the
  # harnesses' idle gate, which samples before the rep, cannot see it.
  #
  # `-x` on the PROCESS NAME, not `-f` on the command line. `-f inkling_forward`
  # reported THREE on an idle-but-for-one-run box, because `bench-decode.sh`
  # carries `--bin target/release/inkling_forward` in its own argv and so does
  # the `sh -c` above it -- so the harness counted its own wrappers as decode
  # processes. An instrument whose false positive is "somebody else is here"
  # would have convicted the wrong thing, and it did, for one reading, before
  # a `ps` said otherwise.
  if [ $((TICK % APPS_EVERY)) -eq 1 ]; then
    INKPIDS=$(pgrep -c -x 'inkling_forward' 2>/dev/null || echo 0)
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t' \
    "$TS_MS" "$GU" "$GMU" "$SMC" "$MEMC" "$TEMP" "$PWR" "${THR:--}" "$GMEM" "$APPS"
  printf '%s\t%s\t%s\t%s\t%s\t' \
    "${MI[MemAvailable]:--}" "${MI[MemFree]:--}" "${MI[AnonPages]:--}" \
    "${MI[AnonHugePages]:--}" "$(( ${MI[SwapTotal]:-0} - ${MI[SwapFree]:-0} ))"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t' \
    "${VS[thp_fault_alloc]:--}" "${VS[thp_collapse_alloc]:--}" "${VS[thp_split_page]:--}" \
    "${VS[compact_stall]:--}" "${VS[pgmajfault]:--}" "${VS[numa_pages_migrated]:--}"
  printf '%s\t%s\t%s\t%s\t%s\t' \
    "$PSI_CPU" "$PSI_MEM_S" "$PSI_MEM_F" "$PSI_IO_S" "$PSI_IO_F"
  printf '%s\t%s\t%s\t%s\n' "$L1" "$PR" "$CPUKHZ" "$INKPIDS"

  if [ "$DURATION" != 0 ] && [ $(( $(date +%s) - START )) -ge "$DURATION" ]; then
    break
  fi
  sleep "$INTERVAL"
done >> "$OUT"
