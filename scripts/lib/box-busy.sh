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
# 5. SOURCED FROM THE WRONG SHELL. zsh does not set BASH_SOURCE, so the
#    run-directly guard at the bottom (`[ "${BASH_SOURCE[0]:-$0}" = "$0" ]`)
#    fires even when sourced -- and the script then self-matches on local
#    processes, which is failure 1 wearing a shell-portability costume.
#    -> Source it under bash: `bash -c '. scripts/lib/box-busy.sh; wait_until_idle 600 host'`.
#    The guard below now also requires BASH_VERSION, so a zsh source is inert
#    rather than wrong.
#
# 6. THE PATTERN INSIDE AN ENVIRONMENT ASSIGNMENT. `pgrep -f cargo` matches a
#    tmux launcher whose argv carries `env PATH=/home/x/.cargo/bin:...`, so a
#    box with an agent session open on it read as busy forever (2026-09-03:
#    the exact-source runner refused sky three times running). -> The pattern
#    is matched against the BASENAMES of argv words, anchored at the start,
#    with `NAME=value` words dropped: a workload is a program, and a program
#    is named by a word of its own, not by a substring of someone's PATH.
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
  for pid in $(ps -eo pid= 2>/dev/null); do
    case " $mine " in *" $pid "*) continue ;; esac   # never ourselves
    local cmd
    cmd=$(ps -o command= -p "$pid" 2>/dev/null) || continue
    [ -n "$cmd" ] || continue
    # Failure 6: match program NAMES, not command-line text. Each argv word
    # that is not an environment assignment is reduced to its basename and the
    # pattern must match it from its first character; `python -m sglang.x`
    # still matches on its module word, `env PATH=...cargo/bin... claude` does
    # not.
    # The PROGRAM is the first word that is not an assignment (`env A=1 prog`
    # execs prog, so the running process shows prog first anyway), plus the
    # module after a `-m` when the program is python, which is how the
    # sglang/vllm servers are named. Nothing else in argv names a workload: a
    # runner whose argv says which cargo command it will launch later is not
    # cargo yet, and this script's own `--remote` caller carries the whole
    # alternation as one word.
    local word program="" module="" prev="" matched=0
    for word in $cmd; do
      if [ -z "$program" ]; then
        case "$word" in *=*) continue ;; esac
        program=${word##*/}
      elif [ "$prev" = "-m" ]; then
        module=$word; break
      fi
      prev=$word
    done
    for word in "$program" "$module"; do
      [ -n "$word" ] || continue
      if [[ $word =~ ^($pat)([^A-Za-z0-9_]|$) ]]; then matched=1; break; fi
    done
    [ "$matched" = 1 ] || continue
    # A SEARCHER IS NOT A WORKLOAD. Failure 1 says we must not match OURSELVES,
    # and the ancestry filter above does that exactly. It does not cover someone
    # ELSE'S searcher: a process whose whole job is to grep for the pattern
    # carries the pattern in its own argv, so it reads as the workload it is
    # waiting for. Live example, 2026-08-27 on the tail box, two of them:
    #
    #   bash -c until ! pgrep -f "cargo build --release --bin w4a16_swz_probe"; do sleep 20; done; ... inkling_forward ...
    #
    # matched here because its argv mentions `inkling_forward`, and reported the
    # box busy for every gated measurement on it -- indefinitely, since that
    # loop also self-matches and can never exit.
    #
    # This CANNOT hide a real measurement, which is why it is safe to drop.
    # `env A=1 prog args` EXECS prog, so the process actually running a
    # measurement always appears in its own right with its own argv; excluding a
    # shell wrapper that merely mentions `pgrep` never removes the workload
    # itself from the list. The change can only make this gate report busy LESS
    # often, and only for processes that are searching rather than running.
    case "$cmd" in *pgrep*) continue ;; esac
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
if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
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
