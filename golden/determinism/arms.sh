#!/bin/bash

# No baked-in defaults: a path that silently points at one machine's home
# directory measures the wrong thing everywhere else, and leaks whose machine
# it was. Set these, or the harness refuses to run.
need() {
  eval "v=\${$1:-}"
  if [ -z "$v" ]; then
    echo "$1 is not set. This harness has no default; set it to this machine's path." >&2
    exit 2
  fi
}
need PILE_PATH
# The run-to-run nondeterminism, reproduced on demand and attributed.
#
#   arms.sh <nruns> <hogGiB>
#
# Four cells. The two CONTROLS establish that aliasing and copying are the same
# arithmetic on the same bytes; the two ARMS run under host memory-reclaim
# pressure and separate "the arithmetic is unordered" from "the operand was not
# there when the kernel read it".
#
#              no pressure          under pressure
#   ALIASED    canonical            WRONG, differently every time
#   COPIED     canonical (same)     canonical
#
# The mechanism: expert weights are not uploaded. `Aliases::register` hands the
# GPU the pile's own `mmap` and cubecl's CUDA backend does nothing to it -- GB10
# reports `pageableMemoryAccessUsesHostPageTables = 1`, so a kernel just
# dereferences the host address. The pile is 159 GiB of FILE-BACKED mapping on a
# 119 GiB box, and file-backed page-cache pages are clean and reclaimable. When
# the kernel reclaims one, nothing tells the GPU: cubecl's own comment on
# `register_external_aliased` says it outright -- "a kernel reading unmapped
# pages does not reliably fault, it reads whatever the address now holds".
#
# So the failure needs memory pressure, which is why it looked random: on an
# idle box the pages stay resident and the runtime is bitwise deterministic.
#
# `INK_STARTUP_COPY=0` is the deliberately unsafe diagnostic arm. The fixed
# lane is the default and aliases the anonymous startup copy.
set -u
N=${1:-3}
HOG=${2:-50}
PILE="$PILE_PATH"
BIN="$FWD_BIN"
IDS=${IDS_PATH:?set IDS_PATH to a prompt .ids file}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${OUT_DIR:-/tmp/ink_determinism}

cleanup() { [ -n "${HP:-}" ] && kill $HP 2>/dev/null; }
trap cleanup EXIT INT TERM

# run_one <dir> <file alias 0|1> <hog GiB, or 0 for none>
run_one() {
    local D=$1 FILE_ALIAS=$2 HG=$3
    mkdir -p "$D"
    HP=
    if [ "$HG" != "0" ]; then
        timeout 900 python3 "$HERE/hog.py" "$HG" 850 > "$D/hog.log" 2>&1 &
        HP=$!
        # Wait for the hog to have actually FAULTED the memory in. Sleeping a
        # guessed number of seconds here would sometimes measure an unpressured
        # box and call it a pressured one.
        for _ in $(seq 1 300); do grep -q held "$D/hog.log" 2>/dev/null && break; sleep 1; done
    fi
    local copy_env=()
    [ "$FILE_ALIAS" = "1" ] && copy_env=(INK_STARTUP_COPY=0)
    # A timeout on the run itself: under pressure this can OOM, and an OOM that
    # hangs is indistinguishable from one that is merely slow.
    env "${copy_env[@]}" INK_LAYERS=${INK_LAYERS:-0:20} INK_DUMP_DIR="$D" timeout 800 \
        "$BIN" "$PILE" "$IDS" "$D/top5.bin" > "$D/log" 2>&1
    local rc=$?
    [ -n "$HP" ] && { kill $HP 2>/dev/null; wait $HP 2>/dev/null; HP=; }
    printf "  rc=%-4s h02=%-13s h19=%-13s bind_ms=%-9s %s\n" \
        "$rc" \
        "$(sha256sum "$D/h_after_02.bin" 2>/dev/null | cut -c1-12)" \
        "$(sha256sum "$D/h_after_19.bin" 2>/dev/null | cut -c1-12)" \
        "$(grep -oP 'bind \+ enqueue\s+\K[0-9.]+' "$D/log" | head -1)" \
        "$(python3 "$HERE/nanscan.py" "$D" 2>/dev/null)"
    rm -f "$D"/h_after_0[3-9].bin "$D"/h_after_1[0-8].bin
}

rm -rf "$BASE"; mkdir -p "$BASE"
echo "=== CONTROL: no pressure, ALIASED ==="
run_one "$BASE/ctl_alias" 1 0
echo "=== CONTROL: no pressure, STARTUP COPY ==="
run_one "$BASE/ctl_copy" 0 0
echo "=== ARM A: ${HOG} GiB pressure, ALIASED (zero-copy out of the mmap) ==="
for i in $(seq 1 "$N"); do run_one "$BASE/a_alias_$i" 1 "$HOG"; done
echo "=== ARM B: ${HOG} GiB pressure, STARTUP COPY (anonymous alias) ==="
for i in $(seq 1 "$N"); do run_one "$BASE/b_copy_$i" 0 "$HOG"; done
