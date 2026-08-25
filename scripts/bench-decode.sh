#!/usr/bin/env bash
#
# bench-decode.sh -- an IDLE-GATED, interleaved decode harness.
#
# The problem it exists for: this repo's rep-to-rep spread on a decode arm is
# 3-7%, which is the same order as several effects that have been reported here
# as real. An ungated number is not a measurement of the change, it is a
# measurement of the change PLUS whatever else had the GPU, and the two cannot
# be separated after the fact. The reference implementation's own tuning log is
# blunt about it: "do not trust any number in this repo that was taken without
# that gate."
#
# So this script refuses to produce a number it cannot stand behind, and it
# labels every number it does produce with the framing that makes it evidence.
#
# WHAT IT DOES
#
#   1. Gates on an IDLE GPU, sampled over a window rather than once -- a single
#      instantaneous sample cannot see a bursty neighbour. It names what is
#      contending. A compute process belonging to ANOTHER USER is a hard refusal
#      that --allow-busy does not lift.
#   2. INTERLEAVES the arms: rep 1 of every arm, then rep 2 of every arm. Drift
#      over the run -- clocks, thermals, page cache, another tenant arriving --
#      lands on every arm equally instead of on whichever went last.
#   3. Discards the COLD passes. `INK_REPEAT=1` is set by default (the project
#      rule) so every step re-runs the identical pass, and the binary's own
#      WARM figures already exclude its first COLD_DECODE_STEPS=2 passes; this
#      script re-derives the same median from the per-step lines and cross-checks
#      the two, so a change to either definition shows up as a disagreement
#      rather than as a silently different number.
#   4. Reports PEAK and MEDIAN separately. The peak is the best rep, which is
#      what a machine can do; the median is what it does. Only the median is
#      honest about a run, and the mean is not reported at all because a single
#      contended rep moves it and it hides that it did.
#   5. Emits the FRAMING RULE with every number -- layer range, context length,
#      what varied, rep count, what was discarded, box, GPU, commit. A figure
#      without its framing is a claim whose evidence has been discarded.
#   6. Checks the governing identity `tok/s = E * 1000 / step_ms` (E = tokens
#      per pass) to 1% on every throughput figure it prints, and says so loudly
#      when it does not hold. Three numbers that do not close are not three
#      numbers, they are a parse error or a population mismatch.
#   7. Gates on BUILD PROVENANCE, which is the same class of failure as the
#      idle gate and has already bitten this project twice. It refuses unless
#      the tree the binary was built in is the tree the script is being run
#      from, at the same commit; it refuses if any source file is newer than
#      the binary, because a binary that predates its source is measuring
#      whatever was there last time; and it records the binary's sha256, size
#      and mtime beside every result, so a stale binary cannot be mistaken for
#      a rebuilt one. On a box where several agents share a checkout, a
#      `git checkout` under your feet turns an A/B into two runs of somebody
#      else's commit and NOTHING in the output would have said so. When two
#      arms name different binaries it also refuses if the two are
#      byte-identical -- a two-arm A/B of one binary against itself has
#      happened here, and it produced a difference.
#
# USAGE
#
#   scripts/bench-decode.sh [options] ARM [ARM ...]
#
#   An ARM is `name:ENV=V ENV=V ...` -- the name, a colon, then the environment
#   that arm varies. The colon is the FIRST one on the line, so `INK_LAYERS=0:21`
#   inside an arm is fine.
#
#     scripts/bench-decode.sh -n 3 base: 'tuned:INK_GEMM_AUTOTUNE=1'
#
#   Options:
#     -n N            reps per arm (default 3)
#     --cold N        decode passes discarded per rep (default 2, = the binary's
#                     own COLD_DECODE_STEPS)
#     --gen N         INK_GEN, decode steps per rep (default 12)
#     --layers R      INK_LAYERS (default 0:21). Recorded in the framing rule.
#     --bin PATH      the binary (default target/release/inkling_forward)
#     --env 'K=V ..'  environment applied to EVERY arm, before the arm's own
#     --no-repeat     do not set INK_REPEAT=1 (you are measuring a growing
#                     sequence on purpose; say so in the notes)
#     --note TEXT     free text carried into the framing rule
#     --util-max N    max GPU utilisation percent to call idle (default 5)
#     --load-max N    max 1-minute loadavg to call the host idle (default 2.0)
#     --samples N     idle samples, 1 s apart (default 3)
#     --allow-busy    downgrade OUR OWN contention to a warning. Never lifts the
#                     other-user refusal, and stamps every result UNGATED.
#     --allow-stale   downgrade the build-provenance refusals to warnings, and
#                     stamp every result STALE. Does not lift the two-arms-are-
#                     the-same-binary refusal.
#     --out DIR       where the logs and the TSV go (default /tmp/bench-decode-<ts>)
#     --gate-only     run the gate, print what it found, exit
#     --              everything after this is passed to the binary (pile,
#                     prompt ids, output path)
#     -h|--help       this text
#
#   An arm may name its own binary with `BENCH_BIN=<path>` inside its env
#   string, which is how you A/B two BUILDS rather than two environments:
#
#     scripts/bench-decode.sh -n 3 'before:BENCH_BIN=/tmp/a/inkling_forward' \
#                                  'after:BENCH_BIN=/tmp/b/inkling_forward'
#
# A WORKED CALL
#
#   scripts/bench-decode.sh -n 3 --gen 16 --layers 0:21 \
#     base: 'tuned:INK_GEMM_AUTOTUNE=1' \
#     -- ~/converted/inkling-small-complete.pile /tmp/prompt.ids /tmp/out.bin
#
# WHERE IT RUNS: on the box with the GPU. It reads `nvidia-smi` locally.

set -uo pipefail

REPS=3
COLD=2
GEN=12
LAYERS="0:21"
BIN="target/release/inkling_forward"
COMMON_ENV=""
REPEAT=1
NOTE=""
UTIL_MAX=5
LOAD_MAX=2.0
SAMPLES=3
ALLOW_BUSY=0
ALLOW_STALE=0
GATE_ONLY=0
OUT=""
ARMS=()
PASSTHRU=()

die() { printf '\n!! %s\n\n' "$*" >&2; exit 2; }

# gawk, not any awk: the parsing below uses 3-argument `match` and `asort`, and
# a mawk box would not fail loudly enough for a measurement script.
AWK=$(command -v gawk || command -v awk)
"$AWK" 'BEGIN{x[1]=1; if (asort(x) != 1) exit 1}' </dev/null 2>/dev/null \
  || die "this script needs gawk (3-arg match, asort). Found: $AWK"

usage() { sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//;$d'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    -n) REPS=$2; shift 2;;
    --cold) COLD=$2; shift 2;;
    --gen) GEN=$2; shift 2;;
    --layers) LAYERS=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    --env) COMMON_ENV=$2; shift 2;;
    --no-repeat) REPEAT=0; shift;;
    --note) NOTE=$2; shift 2;;
    --util-max) UTIL_MAX=$2; shift 2;;
    --load-max) LOAD_MAX=$2; shift 2;;
    --samples) SAMPLES=$2; shift 2;;
    --allow-busy) ALLOW_BUSY=1; shift;;
    --allow-stale) ALLOW_STALE=1; shift;;
    --gate-only) GATE_ONLY=1; shift;;
    --out) OUT=$2; shift 2;;
    -h|--help) usage;;
    --) shift; PASSTHRU=("$@"); break;;
    -*) die "unknown option $1 (try --help)";;
    *) ARMS+=("$1"); shift;;
  esac
done

# ---------------------------------------------------------------------------
# THE GATE
#
# Every refusal here names WHAT is contending, because "the GPU is busy" is not
# actionable and the person who has to act is usually the person who left the
# thing running.
# ---------------------------------------------------------------------------

GATE_FINDINGS=()
GATE_CLK="?"
GATE_HARD=0     # another user, or something we must not shove aside
GATE_SOFT=0     # our own contention: --allow-busy may proceed, loudly

gate() {
  command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi: this script gates on a GPU it can see"

  # -- what the driver says the GPU is doing, over a WINDOW ------------------
  # `memory.used` is [N/A] on unified-memory parts (GB10), which is not a
  # failure and must not be read as 0.
  #
  # And utilisation is NOT sufficient on its own, measured: on 2026-08-25 this
  # gate sampled 0% three times a second apart on a GB10 that had a live
  # `inkling_forward` holding a CUDA context the whole time. A decode process
  # spends long stretches in host-side phases and the counter is coarse, so a
  # gate built on `utilization.gpu` alone would have said "idle" and handed back
  # a poisoned number. The process list is what caught it. That is why this
  # function has four signals and refuses on any of them.
  local util_max_seen=0 mem_note="" i line util mem clk clk_lo="" clk_hi=""
  for ((i = 0; i < SAMPLES; i++)); do
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm --format=csv,noheader 2>/dev/null | head -1)
    util=$(printf '%s' "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print ($1==""?0:$1)}')
    mem=$(printf '%s' "$line" | awk -F, '{gsub(/^ +| +$/,"",$2); print $2}')
    clk=$(printf '%s' "$line" | awk -F, '{gsub(/[^0-9]/,"",$3); print ($3==""?0:$3)}')
    [ "$util" -gt "$util_max_seen" ] 2>/dev/null && util_max_seen=$util
    { [ -z "$clk_lo" ] || [ "$clk" -lt "$clk_lo" ]; } 2>/dev/null && clk_lo=$clk
    { [ -z "$clk_hi" ] || [ "$clk" -gt "$clk_hi" ]; } 2>/dev/null && clk_hi=$clk
    case "$mem" in
      *N/A*) mem_note="memory.used is [N/A] on this part (unified memory); the gate is utilisation and process list only";;
      *) mem_note="memory.used $mem";;
    esac
    [ $((i + 1)) -lt "$SAMPLES" ] && sleep 1
  done
  GATE_FINDINGS+=("GPU utilisation, worst of $SAMPLES samples 1 s apart: ${util_max_seen}%  ($mem_note)")
  # Clocks are part of the framing, not trivia. This part idles near 200 MHz and
  # ramps under load, so a run that starts cold is measuring the ramp in its
  # early passes -- which is most of what the cold discard is for, and it is
  # worth being able to see that the discard had something to do.
  GATE_CLK="${clk_lo:-?}-${clk_hi:-?} MHz"
  GATE_FINDINGS+=("GPU SM clock across those samples: $GATE_CLK")
  if [ "$util_max_seen" -gt "$UTIL_MAX" ]; then
    GATE_FINDINGS+=("  CONTENDED: ${util_max_seen}% > --util-max ${UTIL_MAX}%")
    GATE_SOFT=1
  fi

  # -- who holds the GPU for COMPUTE ----------------------------------------
  # Graphics clients (a compositor, an X server) show up in pmon as type G and
  # are not what poisons a decode timing; a compute client is. Both are listed,
  # only compute refuses.
  local me apps pid pname pmem owner cmd
  me=$(id -un)
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  if [ -n "$apps" ]; then
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      pid=$(printf '%s' "$line" | awk -F, '{gsub(/ /,"",$1); print $1}')
      pname=$(printf '%s' "$line" | awk -F, '{gsub(/^ +| +$/,"",$2); print $2}')
      pmem=$(printf '%s' "$line" | awk -F, '{gsub(/^ +| +$/,"",$3); print $3}')
      owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      cmd=$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-100)
      GATE_FINDINGS+=("COMPUTE process on the GPU: pid $pid  user ${owner:-?}  $pmem  $pname")
      GATE_FINDINGS+=("    $cmd")
      if [ -n "$owner" ] && [ "$owner" != "$me" ]; then
        GATE_FINDINGS+=("  REFUSING: pid $pid belongs to '$owner', not to '$me'. Another user's")
        GATE_FINDINGS+=("            work is not ours to measure around and not ours to displace.")
        GATE_HARD=1
      else
        GATE_SOFT=1
      fi
    done <<< "$apps"
  else
    GATE_FINDINGS+=("no compute processes on the GPU")
  fi

  local gfx
  gfx=$(nvidia-smi pmon -c 1 2>/dev/null | awk '$3=="G"{print "    pid "$2"  "$NF}')
  [ -n "$gfx" ] && GATE_FINDINGS+=("graphics clients present (not a compute contender, listed for the record):" "$gfx")

  # -- ours, whether or not the driver has noticed them yet ------------------
  # A process that has just started, or one that is between kernels, is invisible
  # to --query-compute-apps and will still ruin the run. The name list is the
  # things in this repo that take the GPU for minutes at a time.
  local mine
  mine=$(pgrep -a -f 'inkling_forward|inkling_[a-z0-9_]*gate|inkling_bf16_gemm_bench|inkling_membw|inkling_device_ceiling|nsys|ncu|cuda-gdb' 2>/dev/null \
         | grep -v "^$$ " | grep -v 'bench-decode.sh' | grep -v 'pgrep -a -f')
  if [ -n "$mine" ]; then
    GATE_FINDINGS+=("measurement-shaped processes already running:")
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      pid=${line%% *}
      owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      # Truncated: a rustc invocation is 3 kB of -L flags and the pid and the
      # program are the whole point of the line.
      GATE_FINDINGS+=("    [$owner] $(printf '%.140s' "$line")")
      if [ -n "$owner" ] && [ "$owner" != "$me" ]; then GATE_HARD=1; else GATE_SOFT=1; fi
    done <<< "$mine"
  fi

  # -- the host ------------------------------------------------------------
  # On a unified-memory part the host and the GPU share the memory system, so a
  # busy CPU is a slow GEMM. This is a real contender, not a courtesy check.
  local load busy
  load=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
  GATE_FINDINGS+=("host 1-min loadavg: $load  (max $LOAD_MAX)")
  busy=$(awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{print (l+0 > m+0) ? 1 : 0}')
  if [ "$busy" = "1" ]; then
    GATE_FINDINGS+=("  CONTENDED: loadavg $load > --load-max $LOAD_MAX; on a unified-memory part the host shares the memory system with the GPU")
    GATE_SOFT=1
  fi
}

print_gate() { printf '%s\n' "--- idle gate ---" "${GATE_FINDINGS[@]}"; }

gate
GATED="gated"
if [ "$GATE_HARD" = "1" ]; then
  print_gate >&2
  die "REFUSING TO MEASURE: another user's process holds this GPU. --allow-busy does not lift this."
fi
if [ "$GATE_SOFT" = "1" ]; then
  if [ "$ALLOW_BUSY" = "1" ]; then
    GATED="UNGATED (--allow-busy)"
    print_gate
    printf '\n!! --allow-busy: proceeding on a CONTENDED box. Every number below is stamped\n'
    printf '!! UNGATED and is not comparable with a gated one. Do not quote it as evidence.\n\n'
  else
    print_gate >&2
    die "REFUSING TO MEASURE: the box is not idle (see above). Wait, or fix it, or --allow-busy and wear the UNGATED stamp."
  fi
fi

[ "$GATE_ONLY" = "1" ] && { print_gate; echo; echo "gate: clear"; exit 0; }
[ ${#ARMS[@]} -eq 0 ] && die "no arms given. Try: $0 -n 3 base: 'tuned:INK_GEMM_AUTOTUNE=1'"

# ---------------------------------------------------------------------------
# BUILD PROVENANCE
#
# The idle gate answers "was the box quiet?". This answers the question that
# comes first and is asked less often: "is the binary the code I think it is?"
# Both failures are silent, both produce a plausible number, and this one is
# worse because the number is reproducible -- it will come out the same on the
# next run and on the run after that, and it is still measuring the wrong tree.
#
# Two recorded ways it has gone wrong on this project:
#   * A shared checkout with several agents in it, where one agent's
#     `git checkout` reverted another's between two commands. The second agent
#     built, benchmarked and reported a commit it never wrote.
#   * A two-arm A/B where both arms were byte-identical binaries. It reported a
#     difference, because two runs of the same thing always do.
# ---------------------------------------------------------------------------

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
mtime_of() { date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || stat -c %y "$1"; }

PROV_HARD=0
declare -A BIN_SHA BIN_MTIME

RUN_TREE=$(git -C "$(cd "$(dirname "$0")" && pwd)" rev-parse --show-toplevel 2>/dev/null)
RUN_HEAD=$(git -C "${RUN_TREE:-.}" rev-parse HEAD 2>/dev/null || echo unknown)
RUN_DIRTY=$(git -C "${RUN_TREE:-.}" status --porcelain 2>/dev/null | head -1)

provenance() {
  local bin=$1 label=$2 abs tree head dirty newer
  [ -x "$bin" ] || { echo "  !! $label: no executable at $bin"; PROV_HARD=1; return; }
  abs=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
  BIN_SHA[$abs]=$(sha_of "$abs")
  BIN_MTIME[$abs]=$(mtime_of "$abs")
  echo "  $label: $abs"
  echo "      sha256 ${BIN_SHA[$abs]}"
  echo "      built  ${BIN_MTIME[$abs]}   $(wc -c < "$abs" | tr -d ' ') bytes"

  tree=$(git -C "$(dirname "$abs")" rev-parse --show-toplevel 2>/dev/null)
  if [ -z "$tree" ]; then
    echo "      !! not inside a git tree, so nothing can say what it was built from"
    PROV_HARD=1
    return
  fi
  head=$(git -C "$tree" rev-parse HEAD 2>/dev/null)
  dirty=$(git -C "$tree" status --porcelain 2>/dev/null | head -1)
  echo "      tree   $tree"
  echo "      HEAD   ${head:0:12}$([ -n "$dirty" ] && echo '  (DIRTY -- the commit does not identify what ran; the sha256 above does)')"

  # The tree the binary was built in must be the tree this script is being run
  # from. On a box where several checkouts of the same repo exist, "the binary
  # in target/release" is not a location, it is whichever tree you happened to
  # be standing in.
  if [ -n "$RUN_TREE" ] && [ "$tree" != "$RUN_TREE" ]; then
    echo "      !! BUILT IN A DIFFERENT TREE than this script is running from:"
    echo "         script: $RUN_TREE"
    echo "         binary: $tree"
    PROV_HARD=1
  elif [ -n "$RUN_HEAD" ] && [ "$head" != "$RUN_HEAD" ] && [ "$RUN_HEAD" != unknown ]; then
    echo "      !! HEAD MISMATCH: script at ${RUN_HEAD:0:12}, binary's tree at ${head:0:12}"
    PROV_HARD=1
  fi

  # A binary older than its source is a binary of the previous source. This is
  # the cheap check that a rebuild actually happened, and it is the one an
  # A/B skips when it is in a hurry.
  newer=$(find "$tree/src" "$tree/Cargo.toml" "$tree/Cargo.lock" -newer "$abs" -print -quit 2>/dev/null)
  if [ -n "$newer" ]; then
    echo "      !! STALE: $newer is newer than the binary. Rebuild, or you are"
    echo "         measuring the tree as it was before your edit."
    PROV_HARD=1
  fi
}

echo "--- build provenance ---"
echo "  script tree : ${RUN_TREE:-<not a git tree>} @ ${RUN_HEAD:0:12}$([ -n "$RUN_DIRTY" ] && echo ' (DIRTY)')"
declare -A ARM_BIN
for spec in "${ARMS[@]}"; do
  name=${spec%%:*}; aenv=${spec#*:}; [ "$aenv" = "$spec" ] && aenv=""
  abin=$BIN
  for kv in $aenv; do case "$kv" in BENCH_BIN=*) abin=${kv#BENCH_BIN=};; esac; done
  ARM_BIN[$name]=$(cd "$(dirname "$abin")" 2>/dev/null && pwd)/$(basename "$abin")
  provenance "$abin" "arm '$name'"
done

# Two arms, one binary: fine when they differ by ENVIRONMENT, which is the
# normal case here. Not fine when they were given DIFFERENT PATHS, because then
# the whole point was two builds and there is only one.
for a in "${!ARM_BIN[@]}"; do
  for b in "${!ARM_BIN[@]}"; do
    [ "$a" = "$b" ] && continue
    pa=${ARM_BIN[$a]}; pb=${ARM_BIN[$b]}
    [ "$pa" = "$pb" ] && continue
    if [ "${BIN_SHA[$pa]:-x}" = "${BIN_SHA[$pb]:-y}" ]; then
      echo "  !! arms '$a' and '$b' name DIFFERENT paths holding BYTE-IDENTICAL binaries."
      echo "     $pa"
      echo "     $pb"
      die "REFUSING: that A/B compares a binary with itself. It will report a difference and the difference will be noise."
    fi
  done
done

if [ "$PROV_HARD" = "1" ]; then
  if [ "$ALLOW_STALE" = "1" ]; then
    printf '\n!! --allow-stale: proceeding with unverified build provenance. Every number\n'
    printf '!! below is stamped STALE and does not identify the code that produced it.\n\n'
    GATED="$GATED / STALE (--allow-stale)"
  else
    die "REFUSING TO MEASURE: the binary cannot be tied to this tree at this commit (see above). Rebuild here, or --allow-stale and wear the stamp."
  fi
fi
echo
[ -z "$OUT" ] && OUT="/tmp/bench-decode-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT" || die "cannot make $OUT"
TSV="$OUT/results.tsv"
printf 'arm\trep\ttok_s\tstep_ms\tE_warm\tE_all\twarm_steps\tharness_median_ms\tctx\tidentity_err_pct\tbin_sha256\tbin_mtime\tbin\n' > "$TSV"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)

print_gate
echo
echo "--- plan ---"
echo "  arms      : ${ARMS[*]}"
echo "  reps      : $REPS, INTERLEAVED (rep 1 of every arm, then rep 2 of every arm)"
echo "  discarding: the first $COLD decode passes of every rep"
echo "  logs      : $OUT"
echo

# ---------------------------------------------------------------------------
# ONE REP
#
# Parses three things out of the binary and refuses to reconcile them silently:
#   - its WARM lines, which exclude its own cold passes;
#   - the per-step `pass_ms` values, from which this script takes its OWN median
#     after discarding $COLD;
#   - `tokens per pass`, over ALL passes, which is the E in the identity.
# ---------------------------------------------------------------------------

run_rep() {
  local arm_name=$1 arm_env=$2 rep=$3 log abin
  log="$OUT/${arm_name}.rep${rep}.log"
  abin=${ARM_BIN[$arm_name]}
  local envs=()
  [ "$REPEAT" = "1" ] && envs+=("INK_REPEAT=1")
  envs+=("INK_GEN=$GEN" "INK_LAYERS=$LAYERS")
  # shellcheck disable=SC2206
  [ -n "$COMMON_ENV" ] && envs+=($COMMON_ENV)
  # `BENCH_BIN` selects the arm's binary; it is not an environment variable the
  # binary should see, so it does not travel with the rest.
  local kv
  for kv in $arm_env; do
    case "$kv" in BENCH_BIN=*) ;; *) envs+=("$kv");; esac
  done

  printf '  %-12s rep %d ... ' "$arm_name" "$rep"
  {
    printf '# %s\n' "arm=$arm_name env=${envs[*]} bin=$abin sha256=${BIN_SHA[$abin]:-unknown} args=${PASSTHRU[*]:-}"
  } > "$log"
  local t0 t1
  t0=$(date +%s)
  env "${envs[@]}" "$abin" ${PASSTHRU[@]+"${PASSTHRU[@]}"} >> "$log" 2>&1
  local rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ]; then
    printf 'FAILED rc=%d (see %s)\n' "$rc" "$log"
    return 1
  fi

  "$AWK" -v arm="$arm_name" -v rep="$rep" -v cold="$COLD" -v tsv="$TSV" \
       -v bsha="${BIN_SHA[$abin]:-unknown}" -v bmt="${BIN_MTIME[$abin]:-unknown}" -v bpath="$abin" '
    /WARM steps only/      { if (match($0, /\(([0-9.]+) ms\/step over ([0-9]+) steps/, mm)) { step_ms = mm[1]; wsteps = mm[2] } }
    /WARM per TOKEN/       { if (match($0, /\(([0-9.]+) tok\/s over ([0-9]+) tokens/, tt)) { toks = tt[1]; wtokens = tt[2] } }
    /tokens per pass/      { e_all = $NF }
    /pass_ms [0-9]/        { if (match($0, /pass_ms ([0-9.]+)/, pm)) { n++; if (n > cold) v[++k] = pm[1] };
                             if (match($0, /ctx ([0-9]+)/, cx)) ctx = cx[1] }
    END {
      if (step_ms == "" || toks == "") { print "    !! could not parse the WARM lines -- did the run produce decode steps?"; exit 3 }
      if (k > 0) { asort(v); med = (k % 2) ? v[int(k/2)+1] : (v[int(k/2)] + v[int(k/2)+1]) / 2 }
      else med = -1   # every pass was discarded: --cold is >= the passes there were
      e_warm = (wsteps > 0) ? wtokens / wsteps : 0
      # THE GOVERNING IDENTITY. tok/s = E * 1000 / step_ms, on the same
      # population. Three numbers that do not close are not three numbers.
      pred = e_warm * 1000.0 / step_ms
      err  = (toks > 0) ? 100.0 * (pred - toks) / toks : 999
      printf "%.3f tok/s, %.1f ms/step over %d warm passes (harness median %s)\n", toks, step_ms, wsteps, (med < 0 ? "n/a -- --cold discarded every pass" : sprintf("%.1f ms", med))
      # A median over three or four passes is a median in name only. This is a
      # warning and not a refusal because a short run is sometimes exactly what
      # you want -- but it must not be quoted as if it were a distribution.
      if (wsteps < 5)
        printf "    -- note: only %d warm passes in this rep (INK_GEN minus the %d discarded). A median over that is thin; raise --gen before quoting it.\n", wsteps, cold
      if (err > 1 || err < -1)
        printf "    !! IDENTITY FAILS: E*1000/step_ms = %.3f but the binary says %.3f tok/s (%.2f%%).\n       One of the three is from a different population. Do not quote this rep.\n", pred, toks, err
      # A drift between E over all passes and E over the warm ones means the
      # cold passes carried different tokens -- the discard is doing real work
      # and a figure that mixed them would be wrong.
      if (e_all != "" && e_warm > 0 && (e_all/e_warm > 1.01 || e_all/e_warm < 0.99))
        printf "    -- note: tokens/pass is %.3f over all passes and %.3f over the warm ones; the cold discard matters here.\n", e_all, e_warm
      # Cross-check of the two definitions of "warm". On a SINGLE-NODE run these
      # measure the same thing and must agree; on a two-node pipe they do not,
      # by construction -- the per-step line excludes the receive and the WARM
      # figure includes it -- so a pipe run is expected to trip this and the
      # message says so rather than pretending to have found something.
      hm = (med > 0) ? 100.0 * (med - step_ms) / step_ms : 0
      if (hm > 2 || hm < -2)
        printf "    !! the harness median (%.1f ms, first %d discarded) and the binary WARM figure (%.1f ms) disagree by %.1f%%.\n       Single-node: the two definitions of \"warm\" have drifted and the number is not safe.\n       Two-node pipe: expected -- the step line excludes the receive, the WARM figure includes it.\n", med, cold, step_ms, hm
      printf "%s\t%d\t%.4f\t%.3f\t%.4f\t%s\t%d\t%.3f\t%s\t%.3f\t%s\t%s\t%s\n", arm, rep, toks, step_ms, e_warm, e_all, wsteps, med, ctx, err, bsha, bmt, bpath >> tsv
    }
  ' "$log" || printf '    !! parse failed for %s\n' "$log"
  printf '    %ds wall, %s\n' "$((t1 - t0))" "$log"
  return 0
}

# ---- the interleave -------------------------------------------------------
for ((rep = 1; rep <= REPS; rep++)); do
  echo "--- rep $rep of $REPS ---"
  for spec in "${ARMS[@]}"; do
    name=${spec%%:*}
    aenv=${spec#*:}
    [ "$aenv" = "$spec" ] && aenv=""
    run_rep "$name" "$aenv" "$rep"
  done
done

# ---- did the box stay idle? ----------------------------------------------
GATE_FINDINGS=(); GATE_HARD=0; GATE_SOFT=0
gate
if [ "$GATE_HARD" = "1" ] || [ "$GATE_SOFT" = "1" ]; then
  echo
  echo "!! THE BOX DID NOT STAY IDLE. Something arrived during the run:"
  print_gate
  echo "!! The numbers below were taken across that. Treat them as UNGATED."
  GATED="$GATED / UNGATED (contention appeared mid-run)"
fi

# ---- the report -----------------------------------------------------------
echo
echo "=== decode throughput ==="
echo
"$AWK" -F'\t' 'NR>1 {
    n[$1]++; t[$1,n[$1]] = $3; s[$1,n[$1]] = $4; if (ctx[$1]=="") ctx[$1]=$9
    if (t[$1,n[$1]] > peak[$1] || peak[$1]=="") peak[$1] = $3
    if (s[$1,n[$1]] < best[$1] || best[$1]=="") best[$1] = $4
    if ($10 > 1 || $10 < -1) bad[$1]++
    if (!($1 in seen)) { seen[$1] = 1; byord[++seq] = $1 }
  }
  function med(a, cnt,   i, v, j) { for (i=1;i<=cnt;i++) v[i]=a[i]; asort(v); return (cnt%2)?v[int(cnt/2)+1]:(v[int(cnt/2)]+v[int(cnt/2)+1])/2 }
  END {
    printf "  %-14s %5s  %14s %12s  %15s %13s  %8s\n", "arm", "reps", "MEDIAN tok/s", "PEAK tok/s", "MEDIAN ms/step", "BEST ms/step", "spread"
    # In FIRST-SEEN order, which is the order the arms were given. `for (a in n)`
    # is unspecified in awk, and a baseline picked at random is a baseline the
    # reader cannot check.
    for (o = 1; o <= seq; o++) {
      a = byord[o]; cnt = n[a]
      for (i = 1; i <= cnt; i++) { tv[i] = t[a,i]; sv[i] = s[a,i] }
      mt = med(tv, cnt); ms = med(sv, cnt)
      lo = tv[1]; hi = tv[1]; for (i=1;i<=cnt;i++) { if (tv[i]<lo) lo=tv[i]; if (tv[i]>hi) hi=tv[i] }
      spread[a] = 100.0*(hi-lo)/mt
      printf "  %-14s %5d  %14.3f %12.3f  %15.1f %13.1f  %7.1f%%%s\n", a, cnt, mt, peak[a], ms, best[a], spread[a], (bad[a] ? "   <- " bad[a] " rep(s) failed the identity check" : "")
      m_[a] = mt
    }
    print ""
    print "  MEDIAN is the honest figure: it is what the box does. PEAK is the best rep,"
    print "  i.e. what it can do when nothing else happens, and it is the number that"
    print "  flatters a change. A difference smaller than the spread column is not a"
    print "  result -- it is the same measurement twice."
    print ""
    bn = byord[1]; b = m_[bn]
    for (o = 2; o <= seq; o++) {
      a = byord[o]; d = 100.0*(m_[a]-b)/b
      printf "  %s vs %s (median): %+.2f%%%s\n", a, bn, d, \
        ((d < 0 ? -d : d) < (spread[a] > spread[bn] ? spread[a] : spread[bn]) \
          ? "   <- SMALLER THAN THE SPREAD. Not a result." : "")
    }
  }' "$TSV"

echo
echo "=== the framing rule (this is part of the number, not a footnote) ==="
echo "  what varied     : the arms, and nothing else -- ${ARMS[*]}"
echo "  per what        : per DECODE STEP (one pass of the layer range below), and"
echo "                    per SECOND of decode wall. Not per token unless E = 1."
echo "  layer range     : INK_LAYERS=$LAYERS"
echo "  context length  : $("$AWK" -F'\t' 'NR==2{print $9}' "$TSV") tokens at the last step (INK_GEN=$GEN$( [ "$REPEAT" = 1 ] && echo ", INK_REPEAT=1 so the context does not grow"))"
echo "  reps            : $REPS per arm, INTERLEAVED; first $COLD decode passes of each rep discarded"
echo "  statistic       : median over reps; peak reported separately; mean not reported"
echo "  identity        : tok/s = E * 1000 / step_ms checked to 1% on every rep"
echo "  box             : $(hostname)  $GPU_NAME"
echo "  build           : tree $RUN_TREE @ ${RUN_HEAD:0:12}$([ -n "$RUN_DIRTY" ] && echo " (DIRTY -- the commit does not identify what ran)")"
for spec in "${ARMS[@]}"; do
  name=${spec%%:*}; abin=${ARM_BIN[$name]}
  echo "                    arm '$name' -> ${BIN_SHA[$abin]:-unknown}  ${BIN_MTIME[$abin]:-unknown}  $abin"
done
echo "  gate            : $GATED   (util-max ${UTIL_MAX}%, load-max $LOAD_MAX, ${SAMPLES} samples)"
echo "  SM clock        : $GATE_CLK at the post-run gate; this part idles near 200 MHz and ramps"
echo "  started         : $STARTED"
[ -n "$NOTE" ] && echo "  note            : $NOTE"
echo "  raw             : $TSV"
echo
echo "  Against what: an arm is only evidence against the OTHER ARM IN THIS RUN."
echo "  A number from a different run of this script is a different framing unless"
echo "  every line above matches, and a number from anywhere else is not comparable."
