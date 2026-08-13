#!/bin/bash
# One NVFP4 forward of the whole 42-layer stack, as a head+tail pair on ONE box.
#
# `inkling_forward` refuses to run the whole stack in one process, so even a
# single-machine run is two processes over a socket. Over loopback that says
# NOTHING about whether the model fits two nodes — both halves are competing for
# the same 119 GiB here — but it says everything about capability, which is a
# property of the weights and the arithmetic and not of the wire.
#
# The tail binds and the head connects, with no retry on the connect, so the
# tail has to be listening first: this waits for its "pipe: listening" line
# rather than sleeping a guessed number of seconds.
#
#   run_fp4.sh <ids.bin> <outdir> [port]
#
# Writes <outdir>/{top5.bin,tail.log,head.log}. The tail owns the logits, so
# top5.bin and the "after token N" lines both come from tail.log.
set -u
IDS=$1
OUT=$2
PORT=${3:-7654}
PILE=${PILE_PATH:-converted/inkling-small-complete.pile}
BIN=${FWD_BIN:-paired/bin/inkling_forward}
SPLIT=${INK_SPLIT:-20}
NL=${INK_NLAYERS:-42}

mkdir -p "$OUT"
rm -f "$OUT/tail.log" "$OUT/head.log" "$OUT/top5.bin"
# The ids that were actually consumed, kept beside the result. The scorer
# re-reads this and refuses the run if it is not the item's prompt: a result
# directory that has drifted from the prompt it claims to answer is a failure
# this tree has already had, and the cure is to make the claim checkable.
cp "$IDS" "$OUT/prompt.ids"

INK_LAYERS=$SPLIT:$NL INK_PIPE=tail:0.0.0.0:$PORT "$BIN" "$PILE" "$IDS" "$OUT/top5.bin" \
    > "$OUT/tail.log" 2>&1 &
TAIL_PID=$!

for _ in $(seq 1 3000); do
    grep -q "pipe: listening" "$OUT/tail.log" 2>/dev/null && break
    kill -0 $TAIL_PID 2>/dev/null || { echo "tail died before listening"; cat "$OUT/tail.log"; exit 1; }
    sleep 1
done

INK_LAYERS=0:$SPLIT INK_PIPE=head:127.0.0.1:$PORT "$BIN" "$PILE" "$IDS" "$OUT/head_top5.bin" \
    > "$OUT/head.log" 2>&1 &
HEAD_PID=$!

wait $HEAD_PID; HEAD_RC=$?
wait $TAIL_PID; TAIL_RC=$?
echo "head rc=$HEAD_RC tail rc=$TAIL_RC"
exit $(( HEAD_RC | TAIL_RC ))
