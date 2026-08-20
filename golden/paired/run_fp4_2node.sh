#!/bin/bash
# One NVFP4 forward of the whole 42-layer stack as a head+tail pair on TWO boxes.
#
# `run_fp4.sh` puts both halves on one machine over loopback. That was always a
# capability measurement and never a fit measurement -- and since the startup
# copy fix gave the admission gate an honest ceiling (~28 layers / ~109 GiB on a
# 119 GiB box), it is not even runnable any more: 42 layers across two processes
# on one box is now correctly refused. So this splits the pair across the direct
# link.
#
#   tail  (this box)      INK_LAYERS=$SPLIT:42  INK_PIPE=tail:0.0.0.0:PORT
#   head  ($INK_REMOTE)   INK_LAYERS=0:$SPLIT   INK_PIPE=head:$INK_TAIL_ADDR:PORT
#
# The tail keeps the logits, so top5.bin and the "after token N" lines still
# come from tail.log and the scorer is unchanged.
#
# The good property of run_fp4.sh is kept: the head connects with no retry, so
# this waits for the tail's "pipe: listening" line rather than sleeping a
# guessed number of seconds. What is added is that the head is now an ssh, and
# an ssh has two new ways to lie -- it can succeed while the remote command
# fails, and it can leave a process behind. So the remote half runs under its
# own `timeout`, its exit status is carried back by ssh, and the tail is killed
# BY PID when the head fails.
#
#   run_fp4_2node.sh <ids.bin> <outdir> [port]
#
# $INK_EXTRA is passed to BOTH halves verbatim, so an arm of a comparison can be
# selected without a second copy of this script. It has to reach both ends: the
# lanes this selects between are symmetric, and a mismatch shows up as a shape
# assertion rather than as a wrong number, which is the good failure but still a
# failure.
set -u
IDS=$1
OUT=$2
PORT=${3:-7654}

PILE=${PILE_PATH:-${MARY_MODELS:+$MARY_MODELS/inkling-small-complete.pile}}
BIN=${FWD_BIN:-}
REMOTE=${INK_REMOTE:-}
RBIN=${INK_REMOTE_BIN:-}
RPILE=${INK_REMOTE_PILE:-$PILE}
TADDR=${INK_TAIL_ADDR:-}
EXTRA=${INK_EXTRA:-}
[ -n "$PILE" ]   || { echo "no model pile: set PILE_PATH to the .pile itself, or MARY_MODELS to the directory holding inkling-small-complete.pile" >&2; exit 2; }
[ -e "$PILE" ]   || { echo "model pile not found: $PILE" >&2; exit 2; }
[ -n "$BIN" ]    || { echo "no runtime binary: set FWD_BIN to the inkling_forward this measurement should use" >&2; exit 2; }
[ -x "$BIN" ]    || { echo "inkling_forward not executable at $BIN" >&2; exit 2; }
[ -n "$REMOTE" ] || { echo "no head box: set INK_REMOTE to the ssh target that runs layers 0:SPLIT" >&2; exit 2; }
[ -n "$RBIN" ]   || { echo "no remote binary: set INK_REMOTE_BIN to the SAME build on \$INK_REMOTE" >&2; exit 2; }
[ -n "$TADDR" ]  || { echo "no tail address: set INK_TAIL_ADDR to the interface the head should dial (the direct link, not ZeroTier)" >&2; exit 2; }
SPLIT=${INK_SPLIT:-21}
NL=${INK_NLAYERS:-42}
TMO=${INK_TIMEOUT:-1500}
SSH="ssh -n -o BatchMode=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3"

mkdir -p "$OUT"
rm -f "$OUT/tail.log" "$OUT/head.log" "$OUT/top5.bin"
# The ids that were actually consumed, kept beside the result; the scorer
# re-reads this and refuses a run whose prompt is not the item's.
cp "$IDS" "$OUT/prompt.ids"

# The head is a different machine and cannot see this filesystem, so the ids go
# over with the run. Named by CONTENT, so a stale file cannot be mistaken for
# this one and two items cannot collide.
SUM=$(sha256sum "$IDS" | cut -c1-16)
RIDS=/tmp/ink2n_${SUM}.ids
scp -q -o BatchMode=yes "$IDS" "$REMOTE:$RIDS" || { echo "could not ship ids to $REMOTE" >&2; exit 2; }

# The previous item's tail held ~76 GiB of anonymous arena and the kernel does
# not hand it back the instant the process exits. Starting the next tail into
# that window makes the admission gate refuse a range that fits perfectly well
# thirty seconds later -- 1 of 43 runs on the first two-node set, and the
# refusal is CORRECT, which is why the cure belongs here and not in the gate.
# Wait for the memory rather than retrying into the same wall.
NEED=${INK_WAIT_AVAIL_GIB:-95}
AV=0
for _ in $(seq 1 60); do
    AV=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
    [ "$AV" -ge "$NEED" ] && break
    sleep 5
done
echo "MemAvailable ${AV} GiB before start (wanted >= $NEED)" >> "$OUT/tail.log"

timeout $TMO env $EXTRA INK_LAYERS=$SPLIT:$NL INK_PIPE=tail:0.0.0.0:$PORT \
    "$BIN" "$PILE" "$IDS" "$OUT/top5.bin" >> "$OUT/tail.log" 2>&1 &
TAIL_PID=$!

for _ in $(seq 1 3000); do
    grep -q "pipe: listening" "$OUT/tail.log" 2>/dev/null && break
    kill -0 $TAIL_PID 2>/dev/null || { echo "tail died before listening"; cat "$OUT/tail.log"; exit 1; }
    sleep 1
done

$SSH "$REMOTE" "timeout $TMO env $EXTRA INK_LAYERS=0:$SPLIT INK_PIPE=head:$TADDR:$PORT '$RBIN' '$RPILE' '$RIDS' /tmp/ink2n_${SUM}.head.bin" > "$OUT/head.log" 2>&1 &
HEAD_PID=$!

wait $HEAD_PID; HEAD_RC=$?
[ $HEAD_RC -ne 0 ] && kill $TAIL_PID 2>/dev/null
wait $TAIL_PID; TAIL_RC=$?
echo "head rc=$HEAD_RC tail rc=$TAIL_RC  (head on $REMOTE layers 0:$SPLIT, tail here layers $SPLIT:$NL)"
exit $(( HEAD_RC | TAIL_RC ))
