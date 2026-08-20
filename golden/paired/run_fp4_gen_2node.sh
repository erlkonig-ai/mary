#!/bin/bash
# The NVFP4 runtime GENERATING, as a head+tail pair on TWO boxes.
#
# `run_fp4_2node.sh` answers one token per prompt, which is a PREFILL and reads
# no KV cache at all. That is the wrong instrument for any change to the cached
# decode lane -- it would score a run in which the change never executed. This
# is the same two-box pair with `INK_KV=1` and `INK_GEN` steps, so the answer
# depends on the cache the way a real generation does, and
# `gen_divergence.py` can then say at every position whether the BF16 reference
# would have taken the same step.
#
#   run_fp4_gen_2node.sh <ids.bin> <outdir> <n_new> [port]
#
# Environment is `run_fp4_2node.sh`'s, unchanged: PILE_PATH/MARY_MODELS,
# FWD_BIN, INK_REMOTE, INK_REMOTE_BIN, INK_TAIL_ADDR, and INK_SPLIT. Anything
# further in $INK_EXTRA is passed to BOTH halves -- the lane has to be the same
# on the two ends or the shapes disagree rather than the numbers.
set -u
IDS=$1
OUT=$2
GEN=$3
PORT=${4:-7655}

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
[ -n "$TADDR" ]  || { echo "no tail address: set INK_TAIL_ADDR to the interface the head should dial (the direct link)" >&2; exit 2; }
SPLIT=${INK_SPLIT:-21}
NL=${INK_NLAYERS:-42}
TMO=${INK_TIMEOUT:-1800}
SSH="ssh -n -o BatchMode=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3"

mkdir -p "$OUT"
rm -f "$OUT/tail.log" "$OUT/head.log" "$OUT/top5.bin"
cp "$IDS" "$OUT/prompt.ids"

SUM=$(sha256sum "$IDS" | cut -c1-16)
RIDS=/tmp/ink2n_${SUM}.ids
scp -q -o BatchMode=yes "$IDS" "$REMOTE:$RIDS" || { echo "could not ship ids to $REMOTE" >&2; exit 2; }

# The previous item's tail held ~76 GiB of anonymous arena and the kernel does
# not hand it back the instant the process exits. Wait for the memory rather
# than starting into a refusal that would have been an admission thirty seconds
# later.
NEED=${INK_WAIT_AVAIL_GIB:-95}
AV=0
for _ in $(seq 1 60); do
    AV=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
    [ "$AV" -ge "$NEED" ] && break
    sleep 5
done
echo "MemAvailable ${AV} GiB before start (wanted >= $NEED)" >> "$OUT/tail.log"

timeout $TMO env $EXTRA INK_KV=1 INK_GEN=$GEN INK_LAYERS=$SPLIT:$NL INK_PIPE=tail:0.0.0.0:$PORT \
    "$BIN" "$PILE" "$IDS" "$OUT/top5.bin" >> "$OUT/tail.log" 2>&1 &
TAIL_PID=$!

for _ in $(seq 1 3000); do
    grep -q "pipe: listening" "$OUT/tail.log" 2>/dev/null && break
    kill -0 $TAIL_PID 2>/dev/null || { echo "tail died before listening"; cat "$OUT/tail.log"; exit 1; }
    sleep 1
done

$SSH "$REMOTE" "timeout $TMO env $EXTRA INK_KV=1 INK_GEN=$GEN INK_LAYERS=0:$SPLIT INK_PIPE=head:$TADDR:$PORT '$RBIN' '$RPILE' '$RIDS' /tmp/ink2n_${SUM}.head.bin" > "$OUT/head.log" 2>&1 &
HEAD_PID=$!

wait $HEAD_PID; HEAD_RC=$?
[ $HEAD_RC -ne 0 ] && kill $TAIL_PID 2>/dev/null
wait $TAIL_PID; TAIL_RC=$?
echo "head rc=$HEAD_RC tail rc=$TAIL_RC  (head on $REMOTE layers 0:$SPLIT, tail here layers $SPLIT:$NL)"
exit $(( HEAD_RC | TAIL_RC ))
