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
need FWD_BIN
need IDS_PATH
need ND_RUNS
# The loopback head+tail pair, at a split that FITS and at the one that does not.
#
#   pair.sh <split> <hi> <nruns> <tag> [zerocopy]
#
# The harness's paired numbers were taken at 0:20 + 20:42 on one box. That is
# ~147 GiB of weights touched on a 119 GiB machine, so the two halves cannot
# both be resident and the kernel is reclaiming page-cache pages for the whole
# run -- which is exactly the regime in which the GPU reads expert weights out
# of pages that are no longer there (see results/nondeterminism.txt).
#
# 0:10 + 10:20 is the same two processes, the same wire, the same code, at ~73
# GiB, which fits. The tail's logits are meaningless there (it unembeds from
# layer 19) but its DETERMINISM is exactly as meaningful, and that is the
# measurement: if the fitting split is stable and the overflowing one is not,
# the pipe is not the variable, the memory is.
set -u
SPLIT=$1
HI=$2
N=$3
TAG=$4
ZC=${5:-1}
PILE="$PILE_PATH"
BIN="$FWD_BIN"
IDS="$IDS_PATH"
PORT=${PORT:-7659}
BASE="$ND_RUNS/$TAG"
TMO=${TMO:-900}

rm -rf "$BASE"; mkdir -p "$BASE"
zc_env=()
[ "$ZC" = "0" ] && zc_env=(INK_ZEROCOPY=0)

for i in $(seq 1 "$N"); do
    D="$BASE/r$i"; mkdir -p "$D"
    # The tail binds and the head connects, so the tail has to be listening
    # first; wait for its own line rather than sleeping a guessed number of
    # seconds.
    timeout $TMO env "${zc_env[@]}" INK_LAYERS=$SPLIT:$HI INK_PIPE=tail:0.0.0.0:$PORT \
        "$BIN" "$PILE" "$IDS" "$D/top5.bin" > "$D/tail.log" 2>&1 &
    TAIL_PID=$!
    ok=0
    for _ in $(seq 1 $TMO); do
        grep -q "pipe: listening" "$D/tail.log" 2>/dev/null && { ok=1; break; }
        kill -0 $TAIL_PID 2>/dev/null || break
        sleep 1
    done
    if [ $ok -ne 1 ]; then
        echo "run $i: tail never listened"; tail -3 "$D/tail.log"
        kill $TAIL_PID 2>/dev/null; wait $TAIL_PID 2>/dev/null
        continue
    fi
    timeout $TMO env "${zc_env[@]}" INK_LAYERS=0:$SPLIT INK_PIPE=head:127.0.0.1:$PORT \
        "$BIN" "$PILE" "$IDS" "$D/head_top5.bin" > "$D/head.log" 2>&1 &
    HEAD_PID=$!
    wait $HEAD_PID; HRC=$?
    # A head that dies mid-pass leaves the tail blocked on a socket read
    # forever; propagate the death rather than waiting out the timeout.
    [ $HRC -ne 0 ] && kill $TAIL_PID 2>/dev/null
    wait $TAIL_PID; TRC=$?
    printf "run %s head=%s tail=%s  top5=%s\n" "$i" "$HRC" "$TRC" \
        "$(sha256sum "$D/top5.bin" 2>/dev/null | cut -c1-16)"
done

# Anything still alive is a bug in this script, not a result.
pkill -f "INK_PIPE=tail:0.0.0.0:$PORT" 2>/dev/null
exit 0
