#!/bin/bash
# gb10-lock.sh -- ONE advisory lock per GB10 box, taken by EVERY agent that runs
# on them, so two measurements cannot overlap.
#
# WHY THIS EXISTS. 2026-08-27, 01:5x: two agents ran decode processes on the same
# pair of boxes at the same time. Each admits ~101 GiB on a 121 GiB part, so the
# second one did not make the first slower -- it OOM-killed it. The victim's head
# process took SIGKILL (rc=137) on rep 2 and its own settle check printed "still
# 19/114 GiB head... this rep may be paging". Neither agent did anything wrong:
# both checked the box was idle before starting, and both were right at the
# moment they looked. An idle CHECK is not a RESERVATION, and the window between
# looking and launching is exactly long enough to lose a night's measurement.
#
# scripts/frontier-bench.sh already has a lock of this shape, but it is named for
# frontier and only frontier takes it, so it protects frontier from frontier and
# nothing from anything else. This is the same mechanism with the scope it needed.
#
# THE MECHANISM. `mkdir` is atomic on every filesystem that matters: it either
# creates the directory or fails, and two callers cannot both win. The directory
# holds an `info` file naming pid, host, tag and start time, so a holder can be
# identified rather than merely detected.
#
# BREAKING A STALE LOCK, in order of exactness:
#   - the lock records a pid AND the host that pid lives on. If that host is the
#     box we are on and the pid is gone, the lock is broken IMMEDIATELY. That is
#     exact and needs no timeout.
# TAKING THE LOCK OBLIGES YOU TO KEEP BEATING. Silence is indistinguishable
# from death, by construction: a holder doing one long uninterrupted stretch
# without calling `refresh` is treated as crashed once the timeout elapses and
# its reservation is broken WHILE IT IS STILL RUNNING. A 7-rep two-node run is
# ~18 min and a cold build can be 30-40, so a naive holder that takes the lock
# once and works is inside the 90-minute window today, but not by much. The
# second layer catches the consequence -- a breaker's idle gate immediately sees
# the true holder's processes and refuses -- so the failure mode is a lost
# reservation and a wasted slot rather than an OOM. Beat anyway.
#
#   - otherwise (the holder is a pid on the OTHER box, which is normal for a
#     two-node run whose tail is remote) age is all we have, and GB10_LOCK_TIMEOUT_S
#     (default 5400 s / 90 min) applies. Chosen to clear a cold build of
#     mary+burn+cubecl plus ~18 min of measurement, so a healthy run is never
#     broken into, while one crash costs at most one slot rather than the night.
#
# USAGE, and note every call passes paths in a FILE rather than on a command line:
#   gb10-lock.sh take <host> <tag>     -> rc 0 took it, rc 3 someone else holds it
#   gb10-lock.sh release <host> <tag>  -> releases only if <tag> holds it
#   gb10-lock.sh check <host>          -> prints the holder, rc 0 free, rc 3 held
#
# TAKING BOTH BOXES: BOTH OR NEITHER, AND NEVER WAIT WHILE HOLDING HALF.
# A two-node run needs spark and spark2 together. If you take one, fail to get
# the other, and then WAIT while still holding the first, you deadlock against
# any other two-node agent doing the same thing in the opposite order -- A holds
# spark2 waiting for spark, B holds spark waiting for spark2, and neither ever
# yields. Nothing in this script prevents that; it is a property of how you call
# it. So: if the second take fails, RELEASE THE FIRST before you sleep, and ask
# for both again from scratch on the next attempt. Holding half of what you need
# while you wait is the one usage that turns a working lock into a hang.
# (Found by the frontier harness, which hit the half-held state and only escaped
# it because that version exited instead of waiting.)
#
# WHAT THIS DOES NOT DO: THERE IS NO QUEUE. It is mutual exclusion and nothing
# more. A caller refused with rc 3 is not remembered, is not owed the next slot,
# and will not be handed anything when the lock frees -- it has to come back and
# ask again. Two consequences, both seen within an hour of this landing:
#
#   - A waiter that gives up on the first refusal simply does not run. If your
#     work matters, POLL with a bounded backoff and a long ceiling, and report
#     "never got a slot" rather than reporting nothing.
#   - With several agents and long holds, an unlucky waiter can STARVE. Nothing
#     here prevents it. If that starts happening, the fix is a real queue (a
#     ticket file, taken in order), not a longer timeout and not politeness.
#
# It is deliberately this simple because the failure it was built for -- two
# runs overlapping and OOM-killing each other -- needs only exclusion, and a
# queue that is wrong is worse than no queue. Revisit when starvation is
# observed, not before.
#
# NEVER `pkill -f` ON A SHARED BOX, and never kill a lock holder's processes to
# take a lock. If a box is held, WAIT or exit cleanly and say so. The lock tells
# you who to ask, not who to kill.
set -uo pipefail

ACTION=${1:-}; HOST=${2:-}; TAG=${3:-}
: "${GB10_LOCK_TIMEOUT_S:=5400}"

usage() { sed -n '2,40p' "$0"; exit 2; }
[ -z "$ACTION" ] && usage
[ -z "$HOST" ] && usage
case "$ACTION" in take|release) [ -z "$TAG" ] && usage ;; esac

# The payload is a QUOTED heredoc: nothing in it expands locally. Everything it
# needs arrives as a positional argument. An unquoted heredoc would run $$,
# $(date) and $(hostname) on THIS machine and write a lock describing the wrong
# host -- which is worse than no lock, because it looks like one.
# NOTE: no `ssh -n` here. -n redirects stdin from /dev/null, which silently
# eats the heredoc and makes every call return nothing at all -- an empty
# answer that reads as "free" is the most dangerous possible failure for a
# lock, so it is called out rather than just avoided.
timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
  "bash -s -- '$ACTION' '$TAG' '$GB10_LOCK_TIMEOUT_S'" <<'REMOTE'
set -uo pipefail
ACTION=$1; TAG=$2; TIMEOUT=$3
LOCKD="$HOME/gb10/box.lock.d"
INFO="$LOCKD/info"
ME=$(hostname)
NOW=$(date -u +%s)

write_info() {
  printf 'host=%s\ntag=%s\nstart=%s\nbeat=%s\n' "$ME" "$TAG" "$NOW" "$NOW" > "$INFO"
}

case "$ACTION" in
  take)
    mkdir -p "$HOME/gb10"
    if mkdir "$LOCKD" 2>/dev/null; then write_info
      echo "TAKEN $TAG -- you MUST call 'refresh $TAG' at least every ${TIMEOUT}s"
      echo "  or this lock goes stale and another agent may take the box while you run."
      exit 0; fi
    [ -f "$INFO" ] || { echo "HELD by an unidentified holder"; exit 3; }
    hhost=$(sed -n 's/^host=//p' "$INFO")
    htag=$(sed -n 's/^tag=//p' "$INFO");  hst=$(sed -n 's/^beat=//p' "$INFO")
    age=$(( NOW - ${hst:-0} ))
    # THERE IS DELIBERATELY NO PID-LIVENESS TEST, and the reason is worth keeping.
    # The first version had one: "a dead pid on this box is a dead lock, no
    # timeout needed", which reads as the more exact answer. It is the more exact
    # answer to the WRONG QUESTION. The pid it recorded was the transient ssh
    # shell that took the lock, which exits the instant the take returns -- so
    # every lock was born dead and the very next caller broke it. Caught by this
    # script's own self-test, where `take b` cheerfully broke `take a` one second
    # later and reported it as a repair. Liveness here is a HEARTBEAT, not a pid:
    # a holder doing long work calls `refresh` to prove it is still there.
    if [ "$age" -gt "$TIMEOUT" ]; then
      rm -rf "$LOCKD"
      if mkdir "$LOCKD" 2>/dev/null; then write_info
        echo "TAKEN $TAG (broke a stale lock: $htag silent ${age}s > ${TIMEOUT}s)"; exit 0; fi
    fi
    echo "HELD by tag=$htag on $hhost, last beat ${age}s ago -- WAIT or exit, do not kill"
    exit 3
    ;;
  refresh)
    # A long run proves it is alive. Cheap, and it lets the timeout be short
    # enough that a crash costs one slot instead of the night.
    [ -d "$LOCKD" ] || { echo "NOT HELD"; exit 3; }
    htag=$(sed -n 's/^tag=//p' "$INFO" 2>/dev/null)
    [ "$htag" = "$TAG" ] || { echo "REFUSING: held by $htag, not $TAG"; exit 3; }
    sed -i "s/^beat=.*/beat=$NOW/" "$INFO" && echo "BEAT $TAG"; exit 0
    ;;
  release)
    [ -d "$LOCKD" ] || { echo "NOT HELD"; exit 0; }
    htag=$(sed -n 's/^tag=//p' "$INFO" 2>/dev/null)
    if [ "$htag" = "$TAG" ]; then rm -rf "$LOCKD"; echo "RELEASED $TAG"; exit 0; fi
    echo "REFUSING: held by $htag, not $TAG"; exit 3
    ;;
  check)
    [ -d "$LOCKD" ] || { echo "FREE"; exit 0; }
    tr '\n' ' ' < "$INFO" 2>/dev/null; echo
    exit 3
    ;;
  *) echo "usage: gb10-lock.sh take|release|check <host> [tag]"; exit 2 ;;
esac
REMOTE
