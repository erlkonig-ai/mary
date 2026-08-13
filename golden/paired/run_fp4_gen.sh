#!/bin/bash
# The NVFP4 runtime GENERATING, rather than answering one token.
#
# Same loopback head+tail pair as `run_fp4.sh`, plus `INK_GEN` steps with the
# KV cache on. The continuation is what `gen_divergence.py` then hands to the
# BF16 reference for teacher-forced scoring — the runtime writes the path and
# the reference says, at every step, whether it would have taken it.
#
# `INK_KV=1` is deliberate and is not free: it is a different lane from the
# uncached one every gate in this tree uses as an oracle. It is the lane a real
# generation runs in, which is the one whose capability is in question.
#
#   run_fp4_gen.sh <ids.bin> <outdir> <n_new> [port]
set -u
IDS=$1
OUT=$2
GEN=$3
PORT=${4:-7655}
# Neither path is baked in. A default is an opinion about someone else's disk:
# wrong on every machine but the one it was written on, and it fails by looking
# in the guessed place and reporting the model missing rather than the path
# guessed. So the pile comes from $PILE_PATH, or from $MARY_MODELS -- mary's own
# model-directory convention, the same one `src/paths.rs` resolves against --
# and the runtime binary from $FWD_BIN, with a loud stop naming the fix when
# neither is set.
#
# $FWD_BIN has no fallback at all, deliberately: this tree holds more than one
# build of `inkling_forward`, and WHICH build produced a number is part of the
# number. A harness that quietly picks one has already lost the property this
# directory exists to defend.
PILE=${PILE_PATH:-${MARY_MODELS:+$MARY_MODELS/inkling-small-complete.pile}}
BIN=${FWD_BIN:-}
[ -n "$PILE" ] || { echo "no model pile: set PILE_PATH to the .pile itself, or MARY_MODELS to the directory holding inkling-small-complete.pile" >&2; exit 2; }
[ -e "$PILE" ] || { echo "model pile not found: $PILE (from PILE_PATH or MARY_MODELS)" >&2; exit 2; }
[ -n "$BIN" ] || { echo "no runtime binary: set FWD_BIN to the inkling_forward that this measurement should use (cargo build --release --features inkling-cuda --bin inkling_forward)" >&2; exit 2; }
[ -x "$BIN" ] || { echo "inkling_forward not executable at $BIN (from FWD_BIN)" >&2; exit 2; }
SPLIT=${INK_SPLIT:-20}
NL=${INK_NLAYERS:-42}

mkdir -p "$OUT"
rm -f "$OUT/tail.log" "$OUT/head.log"
cp "$IDS" "$OUT/prompt.ids"

INK_GEN=$GEN INK_KV=1 INK_LAYERS=$SPLIT:$NL INK_PIPE=tail:0.0.0.0:$PORT \
    "$BIN" "$PILE" "$IDS" "$OUT/top5.bin" > "$OUT/tail.log" 2>&1 &
TAIL_PID=$!

for _ in $(seq 1 3000); do
    grep -q "pipe: listening" "$OUT/tail.log" 2>/dev/null && break
    kill -0 $TAIL_PID 2>/dev/null || { echo "tail died before listening"; cat "$OUT/tail.log"; exit 1; }
    sleep 1
done

INK_GEN=$GEN INK_KV=1 INK_LAYERS=0:$SPLIT INK_PIPE=head:127.0.0.1:$PORT \
    "$BIN" "$PILE" "$IDS" "$OUT/head_top5.bin" > "$OUT/head.log" 2>&1 &
HEAD_PID=$!

wait $HEAD_PID; HEAD_RC=$?
wait $TAIL_PID; TAIL_RC=$?
echo "head rc=$HEAD_RC tail rc=$TAIL_RC"
exit $(( HEAD_RC | TAIL_RC ))
