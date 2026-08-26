#!/bin/bash
# Is this box busy? -- the one check that keeps being written wrong.
#
# Source it (`. scripts/lib/box-busy.sh`) or run it (`box-busy.sh [--remote HOST]`).
#
# WHY THIS FILE EXISTS. The same bug has now been written SIX times in this
# project, by six different authors, in six different scripts. It has cost:
# a waiter that spun for 7 days 6 hours, three more that spun 12-15 hours each,
# a tail-side idle gate that had never once fired, a head-side gate broken by
# the fix for the tail-side one, and a false "MIXED -- NET/Socket" hardware
# verdict. Every instance was some form of "grep for a pattern in text that
# contains the pattern".
#
# THE FOUR FAILURES, and what this file does about each:
#
# 1. SELF-MATCH. `pgrep -f D_bench.sh` matches the pgrep's OWN command line, so
#    `until ! pgrep -f D_bench.sh; do sleep 20; done` can never terminate --
#    regardless of whether the target ever existed. Running from a script file
#    helps but is not sufficient: an inline `bash -c` wrapper still carries the
#    pattern. -> We filter by PID against our own process ancestry, which is
#    exact and does not care how we were invoked.
#
# 2. QUOTES CROSSING THE WRONG NUMBER OF SHELLS. `ssh host "pgrep -f 'a|b|c'"`
#    has its inner quotes eaten locally, so the REMOTE shell parses the bars as
#    PIPES. Doubling the quotes fixes remote and breaks local, because local
#    crosses one shell fewer. Both mistakes were made here in sequence. -> We
#    never send a pattern through a shell. The script is copied to the remote
#    and executed as a FILE, so the pattern crosses zero shells.
#
# 3. AN UNREACHABLE BOX READING AS IDLE. An ssh that fails prints nothing, and
#    nothing looks exactly like "no processes". A gate that cannot fail is not a
#    gate. -> We demand a sentinel round-trip and FAIL CLOSED (busy) otherwise.
#
# 4. A POLL WITH NO DEADLINE. When the producer dies, "still waiting" and "will
#    wait forever" are indistinguishable, and the waiter holds that pose
#    indefinitely. All four zombies above were this. -> `wait_until_idle`
#    REQUIRES a deadline argument. There is deliberately no default.
#
# Exit codes: 0 = BUSY, 1 = idle, 2 = unknown (treat as busy).

set -uo pipefail

# Default: the processes that mean "a measurement is running on this box".
BOX_BUSY_PATTERN=${BOX_BUSY_PATTERN:-'inkling_forward|inkling_membw|tp_allreduce_probe|nsys|ncu|sglang|vllm'}

# Every PID in our own ancestry, so we can never count ourselves as the workload.
_box_busy_own_pids() {
  local p=$$ out=""
  while [ -n "$p" ] && [ "$p" != "0" ] && [ "$p" != "1" ]; do
    out="$out $p"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
  printf '%s' "$out"
}

# Local check. Prints matching "pid command" lines; exit 0 if any survive.
box_busy_local() {
  local pat=${1:-$BOX_BUSY_PATTERN}
  local mine hits
  mine=$(_box_busy_own_pids)
  # -a is not portable (macOS pgrep lacks it), so take PIDs and read argv per PID.
  hits=""
  local pid
  for pid in $(pgrep -f "$pat" 2>/dev/null); do
    case " $mine " in *" $pid "*) continue ;; esac   # never ourselves
    local cmd
    cmd=$(ps -o command= -p "$pid" 2>/dev/null) || continue
    [ -n "$cmd" ] || continue
    hits="$hits$pid $cmd"$'\n'
  done
  if [ -n "$hits" ]; then printf '%s' "$hits"; return 0; fi
  return 1
}

# Remote check. Copies THIS FILE to the peer and runs it there, so the pattern
# never crosses a shell -- which is what failure 2 is about.
box_busy_remote() {
  local host=$1 pat=${2:-$BOX_BUSY_PATTERN}
  local self=${BASH_SOURCE[0]:-$0}
  # Failure 3: an unanswered box is not an idle box.
  if [ "$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$host" echo __UP__ 2>/dev/null)" != "__UP__" ]; then
    echo "box-busy: $host UNREACHABLE -- reporting BUSY (an unanswered box is not an idle box)" >&2
    return 2
  fi
  local dest="/tmp/.box-busy.$$.sh"
  scp -q -o BatchMode=yes "$self" "$host:$dest" 2>/dev/null || { echo "box-busy: cannot stage on $host" >&2; return 2; }
  # BOX_BUSY_PATTERN travels as an environment assignment, not as shell text.
  ssh -n -o BatchMode=yes "$host" "BOX_BUSY_PATTERN=$(printf '%q' "$pat") bash $dest; rc=\$?; rm -f $dest; exit \$rc"
}

# Block until idle, or give up. THE DEADLINE IS MANDATORY -- see failure 4.
# usage: wait_until_idle <deadline-seconds> [host]
wait_until_idle() {
  local deadline=${1:?wait_until_idle needs a deadline in seconds; a poll without one becomes a zombie}
  local host=${2:-}
  local waited=0 step=15
  while :; do
    if [ -n "$host" ]; then box_busy_remote "$host" >/dev/null; else box_busy_local >/dev/null; fi
    case $? in
      1) return 0 ;;                                  # idle
      2) echo "box-busy: unknown state, treating as busy" >&2 ;;
    esac
    if [ "$waited" -ge "$deadline" ]; then
      echo "box-busy: STILL BUSY after ${deadline}s -- giving up rather than waiting forever." >&2
      echo "box-busy: (if you expected this to clear, the producer may have died; check, do not wait.)" >&2
      return 1
    fi
    sleep $step; waited=$((waited + step))
  done
}

# Run directly: report and exit with the code.
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
  if [ "${1:-}" = "--remote" ]; then
    shift; box_busy_remote "$@"; rc=$?
  else
    box_busy_local "${1:-}"; rc=$?
  fi
  case $rc in
    0) echo "BUSY" >&2 ;;
    1) echo "idle" >&2 ;;
    2) echo "UNKNOWN (treating as busy)" >&2 ;;
  esac
  exit $rc
fi
