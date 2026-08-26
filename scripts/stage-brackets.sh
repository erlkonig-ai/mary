#!/bin/bash
# stage-brackets.sh -- pull the FOUR OUTER BRACKETS out of an inkling_forward
# log and report their WARM medians, per decode step, for that ONE NODE.
#
# WHY THIS EXISTS. `inkling_forward` already prints, every pass, a partition of
# the step that tiles it by construction:
#
#     pass prologue        feed construction, before the embedding
#     layer loop           HOST ENQUEUE of this node's layer range
#     DEVICE, one sync     the stack's device time the host had to WAIT for
#     after the sync       RMS lines, residual, head, argmax, commit, draft,
#                          and -- on a pipe HEAD -- the peer wait
#     UNATTRIBUTED         whatever no line names
#
# The three numbers an end-to-end question actually turns on are in there:
#
#     host enqueue    = layer loop
#     EXPOSED device  = DEVICE-one-sync   (the ONLY device time the step sees)
#     own step        = step - peer wait  (a pipe head's own half)
#
# and nothing else in this repo reports the exposed figure at all. A device-side
# kernel win can only reach the step through the EXPOSED term; whatever share of
# the device work ran while the host was still enqueueing is already hidden and
# a win there buys nothing. That ratio is different in different configurations,
# which is exactly the thing that has to be measured rather than assumed.
#
# FRAMING. Every figure printed is MS PER DECODE STEP, ONE NODE, ONE PROCESS,
# for the layer range and context that log's own header names. It is not per
# token unless tokens/pass is 1, and it is not per layer. The first `--cold`
# warm-eligible passes are discarded (default 2, the binary's own
# COLD_DECODE_STEPS) because they still carry first-touch uploads and kernel
# JIT; a decode pass is additionally only counted when its pass_ms is under
# `--max-ms` (default 1000), which drops the prefill.
#
# USAGE
#   scripts/stage-brackets.sh [--cold N] [--max-ms MS] LOG [LOG ...]
#
# One line per log, plus a per-arm summary when the log names follow the
# pipe-bench convention `<arm>.rep<N>.<node>.log`.
set -u

COLD=2
MAXMS=1000
while [ $# -gt 0 ]; do
  case ${1:-} in
    --cold)   COLD=$2; shift 2 ;;
    --max-ms) MAXMS=$2; shift 2 ;;
    -h|--help) sed -n '1,/^set -u/p' "$0" | sed -e 's/^#\{1,\} \{0,1\}//' -e '/^set -u/d'; exit 0 ;;
    *) break ;;
  esac
done
[ $# -ge 1 ] || { echo "usage: stage-brackets.sh [--cold N] [--max-ms MS] LOG ..." >&2; exit 2; }

printf '%-34s %5s %9s %9s %9s %9s %9s %9s\n' \
  log n step_ms enqueue exposed after_sync peer_wait own_step

for f in "$@"; do
  gawk -v cold="$COLD" -v maxms="$MAXMS" -v name="$(basename "$f")" '
    function med(v, k,   i, w, j) { for (i=1;i<=k;i++) w[i]=v[i]; asort(w);
      return (k%2) ? w[int(k/2)+1] : (w[int(k/2)]+w[int(k/2)+1])/2 }
    /DEVICE, one sync for this node/ { exposed = $NF }
    /^ *layer loop / { enq = $3 }
    /^ *after the sync / { aft = $4 }
    /MTP draft/ { if (match($0, /peer wait *([-0-9.]+)/, m)) peer = m[1]; else peer = 0 }
    # The `step N:` line closes a pass and carries its wall cost.
    /^ *step [0-9]+: / {
      if (match($0, /pass_ms ([0-9.]+)/, m)) {
        p = m[1] + 0
        if (p < maxms && enq != "") {
          seen++
          if (seen > cold) {
            n++; S[n]=p; E[n]=enq+0; X[n]=exposed+0; A[n]=aft+0; P[n]=peer+0
            O[n]=p - peer
          }
        }
      }
      enq=""; exposed=""; aft=""; peer=0
    }
    END {
      if (n == 0) { printf "%-34s %5s   (no warm decode passes)\n", name, "-"; exit }
      printf "%-34s %5d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f\n",
        name, n, med(S,n), med(E,n), med(X,n), med(A,n), med(P,n), med(O,n)
    }' "$f"
done
