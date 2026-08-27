#!/bin/bash
# frontier-bench.sh -- the frontier, measured unattended, recorded in git.
#
# One command, no decisions: fast-forward main on both boxes, build the release
# binary ONCE, run the scoreboard lane on it, append one row to
# `bench/frontier.tsv`, commit, push, and print a verdict against the previous
# row. Safe to run repeatedly, and safe to run beside other people's work --
# see BOX SAFETY, which is the half of this file that matters most.
#
# ---------------------------------------------------------------------------
# WHAT THE NUMBER IS, AND ITS FRAMING RULE
#
# A row is the SCOREBOARD lane's median decode throughput of `main`, as `main`
# runs OUT OF THE BOX. Every figure carries what it is per, at what config, and
# against what:
#
#   PER      one decode step of the 42-layer two-node pipeline -- head layers
#            0:21 on the head box, tail 21:42 on the tail box. NOT per token of
#            a batch, and NOT per layer-step. (The `layer loop` bracket is a
#            different quantity again: ~61% of it is blocked device wait,
#            a3a2d93. Nothing here is built on it. `nsys-bracket.py`'s INK_GEN
#            differencing mode was RETRACTED in cde5de9 and is not used here at
#            all -- this script runs no profiler.)
#   AT       ctx 3732 (`~/refprompts/ctx3732.ids`, md5 pinned below and checked
#            before every run), INK_GEN=64, INK_KV=1, split 21, `--overlap`,
#            `--order fixed`, 7 process-reps; the median over the reps that
#            COMPLETED, and the row records the true n beside the n requested.
#   AGAINST  the previous row of the SAME arm, with the resolution this
#            comparison can see printed beside the delta and stored in the row.
#
# THE FRONTIER ARM SETS NO `INK_*` SWITCH OF ITS OWN. That is the definition:
# the frontier is what main gives you when you run it, not the best arm anyone
# can assemble. So the default-ON wins are in it BECAUSE they are on by default
# -- INK_FUSE_QKVR, INK_DEV_ROUTE, INK_ACT_BF16, INK_DEV_PLAN, INK_SWZ, the
# W4A16 head/sink lanes and NVFP4 KV pages (which have no switch left at all),
# INK_ANN_HEAD=8192, INK_DRAFT_TOPK=512, and the W4A16 depth-4 swizzle lane
# whose `swizzle_pays` predicate decides per shape (`INK_W4A16_SWZ=0` is its
# ablation, and this run does not set it). The default-OFF experimental lanes
# -- INK_GRAPH, INK_GRAPH_LANE, INK_TP -- are OUT of it for exactly the same
# reason, and folding one in would make the series measure a cherry-pick that
# nobody gets by checking out main.
#
# Experimental arms CAN be recorded: `FRONTIER_EXTRA_ARM='name:HEADENV|TAILENV'`
# adds one to the same interleaved run and writes it as its OWN row with
# kind=experimental. It is never folded into the frontier row. SEVERAL arms are
# separated by ';', and side by side in ONE run is the point: arms within a run
# are interleaved and so compare PAIRED, at about this lane's 1.1%, where two
# rows from two sessions do not and resolve far worse. E.g.
#
#   FRONTIER_EXTRA_ARM='graph:INK_GRAPH_LANE=1 INK_GRAPH_CARRY=1|INK_GRAPH_LANE=1 INK_GRAPH_CARRY=1;shuffle:INK_W4A16_SWZ_SHUFFLE=1|INK_W4A16_SWZ_SHUFFLE=1'
#
# WHAT main's SHA DOES NOT PIN, and is therefore recorded per row: mary's
# `[patch]` section points `cubecl-runtime`/`cubecl-cuda`/`cubecl-wgpu` at
# `../cubecl-graph` and `triblespace-core` at `../triblespace-rs`, both by
# RELATIVE PATH. Those two working copies float independently of mary's main, so
# a row naming only main's sha would not identify the binary it measured. Both
# shas go in the row, and a dirty sibling is flagged in `notes`.
#
# ---------------------------------------------------------------------------
# BOX SAFETY
#
# These two boxes are SHARED, and several agents run measurements on them
# concurrently. An automatic benchmark that stomps on a live measurement is
# worse than no benchmark. Four rules, all enforced below:
#
#  1. AN IDLE GATE, self-excluding and fail-closed, run right after the
#     reservation below: the lock stops the NEXT agent, the gate catches one
#     that was already in flight when we reserved. `scripts/lib/box-busy.sh`
#     is the one correct implementation in this tree -- it filters by PID
#     against our own process ancestry (so it cannot match itself, the bug this
#     project has written six times), it copies itself to the peer and runs it
#     as a FILE (so no pattern crosses a shell), and it reports an unreachable
#     box as BUSY. If either box is busy this script EXITS CLEANLY (rc 3) and
#     records nothing. It does not wait for hours: `FRONTIER_WAIT_S` (default 0)
#     is the only wait there and it is bounded (default 300 s, which covers a box
#     that has just been released and is still winding a run down).
#
#  2. AN ADVISORY LOCK ON EACH BOX -- `scripts/gb10-lock.sh`, the SHARED one,
#     taken by every agent that runs on these boxes. This script had its own
#     frontier-named lock first, and the scope was the bug: it protected
#     frontier from frontier and nothing from anything else. On 2026-08-27 two
#     agents launched decode processes on the same pair within seconds of each
#     other; each admits ~101 GiB on a 121 GiB part, so the second did not make
#     the first slower, it OOM-killed it (SIGKILL on rep 2, and the victim's own
#     settle printed "still 19/114 GiB head... this rep may be paging"). Both had
#     checked the box was idle and both were right when they looked. AN IDLE
#     CHECK IS NOT A RESERVATION -- the gate below cannot close that window, and
#     the lock is what does.
#
#     IT LOOKS BEFORE IT RESERVES. A stale lock is only broken on a box that is
#     also visibly idle, because gb10-lock reads silence as death while this
#     caller can actually see the holder still running -- breaking a live
#     holder's reservation wastes a slot and deletes someone's claim for
#     nothing. Same rule covers a box busy with an unlocked run.
#
#     A REFUSAL IS A WAIT, NOT AN EXIT. There is no queue behind the lock, so an
#     agent that walks away on the first refusal is simply never handed the box
#     -- and for a nightly benchmark that means the row does not exist. It polls
#     every FRONTIER_LOCK_POLL_S to a ceiling of FRONTIER_LOCK_WAIT_S (default
#     2 h), logs every refusal with the holder's tag and how long they have had
#     it, gives the head box BACK between attempts (holding half of what you
#     need is how two agents deadlock), and gives up loudly rather than
#     silently. It never overrides a holder: a lock you can be talked out of is
#     not a lock.
#
#     Taken from the CONTROL box, because a box cannot ssh to itself here (host
#     key verification fails) and gb10-lock.sh reaches a box over ssh. Held for
#     the whole run, refreshed every FRONTIER_REFRESH_S while the build and the
#     reps are running, and released in a trap on every exit path. Staleness is
#     SILENCE, not a dead pid: GB10_LOCK_TIMEOUT_S (default 5400 s / 90 min)
#     clears a cold build of mary against burn and cubecl plus ~18 min of
#     measurement, so a healthy run is never broken into while one crash costs a
#     slot rather than the night. The run phase re-reads the lock file on the
#     head box and REFUSES unless its own tag holds it, so a run whose
#     reservation was lost stops instead of measuring unprotected.
#
#  3. NEVER `pkill -f` ON A SHARED BOX. Nothing here kills by pattern. The one
#     place this script kills anything is the run-timeout path, and it matches
#     this run's OWN output path (`pb_out_<TAG>.bin`, and TAG carries a UTC
#     stamp no other run can have), excludes its own shell by PID, and kills by
#     PID. That is the discipline `pipe-bench.sh` learned the hard way on
#     2026-08-26, when a blanket `pgrep -f inkling_forward` reaped reps 3 and 4
#     of a concurrent run and that run's results.tsv reported "5 reps" as though
#     5 had been asked for. Which is also why a short run here is recorded as
#     `short-run:n=<true>-of-<requested>` with whatever the log says about why.
#
#  4. >= 60 s BETWEEN LARGE RUNS. A rep holds ~100 GiB of unified memory and the
#     kernel takes tens of seconds to hand it back, so a run that starts too
#     soon does not get a slower GPU, it gets page faults -- and they land on
#     the warm steps, not on the cold ones the discard removes.
#     `FRONTIER_COOLDOWN_S` (default 60) is slept after the build and before the
#     measurement; `pipe-bench.sh`'s own `settle()` covers rep-to-rep and is a
#     CONDITION on MemAvailable rather than a sleep, so it is usually free.
#
# ---------------------------------------------------------------------------
# HOW IT IS TRIGGERED, AND WHAT IT DOES WHEN NOTHING CHANGED
#
#   frontier-bench.sh --due     exit 0 if a run is warranted, 1 if not.
#                               Cheap, local, and it NEVER touches the network
#                               or the boxes: `orient` re-evaluates habit
#                               conditions every 60 s, so this must cost
#                               milliseconds. It is the same split
#                               `faculties/scripts/stranded-work.sh` uses.
#   frontier-bench.sh           the run.
#   frontier-bench.sh --force   run even when --due says no.
#
# WHEN main HAS NOT MOVED THIS RECORDS A REPEAT, LABELLED kind=repeat, rather
# than skipping. The reason is that the series is otherwise unfalsifiable: at
# n=7 this lane resolves ~1.1% on a PAIRED arm delta inside one interleaved run,
# and a row-to-row comparison is NOT paired, so a later "+1.5%" cannot be told
# from the boxes having drifted unless something in the series measures drift at
# a FIXED sha. A repeat is that measurement, it costs ~18 min, and it skips the
# rebuild (the staged binary is keyed by main + both floating path-deps). It is
# not recorded on every trigger -- only once FRONTIER_HEARTBEAT_H (default 24 h)
# has passed since the last row -- so an idle main costs one run a night, not
# one an hour.
#
# ---------------------------------------------------------------------------
# WHERE IT RUNS. The measurement happens on the HEAD box, because that is where
# the head half of the pipeline and its pile live. This file stages ITSELF and
# box-busy.sh there and re-enters as `--run`; every parameter crosses as an
# environment assignment and never as shell text. The row is committed and
# pushed from the frontier worktree ($FRONTIER_WT under the remote $HOME), which
# is fast-forwarded to origin/main before the run,
# so the commit sits on top of origin/main alone and can never carry another
# agent's unpushed work. That worktree shares the parent repository's hooks, so
# the box's pre-push leak guard audits the push. No `--no-verify`, ever.

set -uo pipefail

# --- parameters -----------------------------------------------------------
# Hosts, users and paths are PARAMETERS, not literals: a committed script that
# bakes in one machine's home directory is both wrong to reuse and a term this
# repo's push hook refuses for the public channel.
#
# The site's actual hosts, user and directory layout live in a MACHINE-LOCAL
# file, sourced here before any default is applied, so nothing site-specific
# has to be committed:
#
#     ~/.frontier-bench.env      (override with FRONTIER_ENV=<path>)
#         FRONTIER_HEAD=<ssh alias of the head box>
#         FRONTIER_TAIL=<ssh alias of the tail box>
#         FRONTIER_TAIL_IP=<the tail, as the HEAD box reaches it>
#         FRONTIER_USER=<remote user>
#         FRONTIER_PARENT=<checkout, relative to the remote $HOME>
#
# It is a plain shell fragment and it is read, not parsed, so keep it to
# assignments. Without it the defaults below assume the simplest possible
# layout ($HOME/mary, same username on both ends), which is right for a fresh
# machine and wrong for most real ones.
# Not in the `--run` phase: there every value arrives from the control box as an
# environment assignment, and a stale env file on the box would silently
# override the parameters the run was actually started with.
FRONTIER_ENV=${FRONTIER_ENV:-$HOME/.frontier-bench.env}
# shellcheck source=/dev/null
[ "${1:-}" != "--run" ] && [ -r "$FRONTIER_ENV" ] && . "$FRONTIER_ENV"
FRONTIER_HEAD=${FRONTIER_HEAD:-spark2}              # head box, as the CONTROL box reaches it
FRONTIER_TAIL=${FRONTIER_TAIL:-spark}               # tail box, as the CONTROL box reaches it
FRONTIER_TAIL_IP=${FRONTIER_TAIL_IP:-10.55.0.1}     # tail box, as the HEAD box reaches it
FRONTIER_USER=${FRONTIER_USER:-$(id -un)}           # remote user on both boxes
FRONTIER_WT=${FRONTIER_WT:-mary-frontier}           # frontier worktree, under remote $HOME
FRONTIER_PARENT=${FRONTIER_PARENT:-mary}            # its parent repository, under remote $HOME
# The path-dep siblings (`../triblespace-rs`, `../cubecl-graph`) are siblings OF
# THE CHECKOUT, so their directory follows from the parent's rather than being a
# second thing to keep in step.
FRONTIER_SIBDIR=${FRONTIER_SIBDIR:-$(dirname "$FRONTIER_PARENT")}
FRONTIER_BINDIR=${FRONTIER_BINDIR:-frontier/bin}    # where the measured binary is staged
FRONTIER_IDS=${FRONTIER_IDS:-refprompts/ctx3732.ids}
FRONTIER_IDS_MD5=${FRONTIER_IDS_MD5:-3f31031d7e44e9a5fcdafd36ebba0217}
FRONTIER_HPILE=${FRONTIER_HPILE:-work-inkling-complete.pile}
FRONTIER_TPILE=${FRONTIER_TPILE:-converted/inkling-small-complete.pile}
FRONTIER_REPS=${FRONTIER_REPS:-7}
FRONTIER_GEN=${FRONTIER_GEN:-64}
FRONTIER_SPLIT=${FRONTIER_SPLIT:-21}
# THE CARGO FEATURES ARE NOT OPTIONAL AND NOT A DETAIL. `inkling_forward`
# declares `required-features = ["inkling-cuda"]`, and the first version of this
# script built it without them. Verified rather than assumed, because the two
# possible behaviours differ enormously and only one of them is survivable:
#
#   cargo build --release --bin inkling_forward
#   error: target `inkling_forward` in package `mary` requires the features: `inkling-cuda`
#
# An explicitly NAMED `--bin` errors and exits nonzero, which this script would
# have reported as a build failure -- a wasted reservation, but a loud one. The
# silent form is `--bins` / `--all-targets`, where cargo SKIPS a target whose
# required features are unmet and exits 0, producing a build that "succeeds"
# with no binary. This uses the named form and additionally demands the artifact
# below, so neither shape can pass for success.
#
# The set matches what every other build of this binary on these boxes uses
# (the boxes' own build.sh, and the other agents' build lines), because a
# frontier row has to be comparable to the project's other scoreboard numbers
# and a differently-featured binary is a different binary. `inkling-cuda`
# already implies `import`; both are named anyway so the row's config column
# reads as the command someone would actually type.
#
# Note what this is NOT: it is not an INK_* switch. Cargo features decide what
# COMPILES, and the frontier's "no switch set" rule is about what the binary
# is TOLD AT RUNTIME. Choosing features so the thing links is not cherry-picking
# an arm.
FRONTIER_FEATURES=${FRONTIER_FEATURES:-inkling-cuda,cuda-backend,import}
GB10_LOCK_TIMEOUT_S=${GB10_LOCK_TIMEOUT_S:-5400}   # gb10-lock.sh reads this
FRONTIER_REFRESH_S=${FRONTIER_REFRESH_S:-300}      # heartbeat while we hold the boxes
FRONTIER_LOCK_WAIT_S=${FRONTIER_LOCK_WAIT_S:-7200} # ceiling on waiting for the reservation
FRONTIER_LOCK_POLL_S=${FRONTIER_LOCK_POLL_S:-180}  # how often to re-ask for it
FRONTIER_LOCK_TAG=${FRONTIER_LOCK_TAG:-}
FRONTIER_COOLDOWN_S=${FRONTIER_COOLDOWN_S:-60}
FRONTIER_RUN_TIMEOUT_S=${FRONTIER_RUN_TIMEOUT_S:-3600}
FRONTIER_HEARTBEAT_H=${FRONTIER_HEARTBEAT_H:-24}
FRONTIER_WAIT_S=${FRONTIER_WAIT_S:-300}   # idle-gate wait AFTER we hold the reservation
FRONTIER_EXTRA_ARM=${FRONTIER_EXTRA_ARM:-}
FRONTIER_PUSH=${FRONTIER_PUSH:-1}
RESULTS_REL=bench/frontier.tsv

# The paired sd of an arm delta on this lane, measured over
# /tmp/pipe-ab_{main,graph}_r1..r7 and recorded in pipe-bench.sh's header.
PAIRED_SD=1.45
# The sd of a SINGLE process-rep of one arm on one box, from the same
# measurement (main 1.40%, graph 1.16%). A row-to-row comparison is UNPAIRED --
# two independent medians from two different runs -- so its resolution is the
# wider one, and that is the one the verdict uses.
UNPAIRED_SD=1.30
# MEASURED, and the series produced it by accident on its first night. The
# first three rows -- 79d3c24, 0c333b3, 60b703b -- all measure decode code
# that is SEMANTICALLY IDENTICAL: `git diff <a> <b> -- src` is zero
# non-comment lines across both hops. They read 11.2850, 11.3460 and 11.2200
# tok/s, a range of 1.12% of the minimum, and the middle one verdicted
# +0.54% while the last verdicted -1.11%, both inside-resolution.
#
# So the EMPIRICAL noise floor across separate runs of unchanged code is
# ~1.12%, and the derived threshold of 1.39% sits just above it. That is the
# validation this number could not otherwise get: a threshold derived from a
# per-rep sd is a model, and three rows of unchanged code is the model being
# checked against the world. Keep taking `kind=repeat` rows -- they are the
# only thing in the series that can separate a real gain from box drift.
#
# ONE EXPECTATION THAT WAS WRONG, recorded so nobody plans around it: a
# comment-only change to src does NOT produce a byte-identical binary. All
# three rows carry different bin_sha256 (55d55aea, 317c341e, 516ec3fe)
# despite zero semantic change, because line numbers reach the binary through
# panic locations and debug info even in release. A drift row cannot be
# identified by matching the binary hash; it has to be identified by diffing
# the source, which is what the paragraph above does.

die() { printf '\n!! %s\n\n' "$*" >&2; exit 2; }
say() { printf '%s\n' "$*"; }

# --- where is the repo ----------------------------------------------------
# Resolved rather than hardcoded, because this file is run three ways: from the
# repo, from the blob the `habit` faculty materializes (whose cwd is the
# directory holding the pile, i.e. the workspace root), and staged into /tmp on
# the head box.
find_repo() {
  local c
  for c in \
      "${FRONTIER_REPO:-}" \
      "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)" \
      "$PWD" "$PWD/mary" "$PWD/../mary"; do
    [ -n "$c" ] || continue
    if [ -f "$c/scripts/pipe-bench.sh" ]; then printf '%s' "$c"; return 0; fi
  done
  return 1
}

iso_to_epoch() {  # GNU first, BSD second, 0 if neither parses it
  date -u -d "$1" +%s 2>/dev/null && return 0
  date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$1" +%s 2>/dev/null && return 0
  echo 0
}

# Newest data row of `bench/frontier.tsv` for one arm, or empty.
# Read from the working tree when it is there, else from the last-fetched
# origin/main -- the row is committed on the head box, so a control box that has
# only fetched (and not merged) still sees the series.
last_row() {  # last_row <repo> <arm>
  local repo=$1 arm=$2 body
  body=$(cat "$repo/$RESULTS_REL" 2>/dev/null)
  [ -n "$body" ] || body=$(git -C "$repo" show "origin/main:$RESULTS_REL" 2>/dev/null)
  printf '%s\n' "$body" | awk -F'\t' -v a="$arm" 'NF>3 && $1!="utc" && $3==a {r=$0} END{if(r!="")print r}'
}

# --- --due ----------------------------------------------------------------
# The CHEAP half, and the contract is that it costs milliseconds: `orient`
# re-evaluates habit conditions every 60 seconds. It does NOT fetch, does NOT
# ssh, and does NOT look at the boxes. It answers only "is there something to
# measure"; the expensive, gated, definitive answer is the run itself.
if [ "${1:-}" = "--due" ]; then
  repo=$(find_repo) || exit 1
  head_sha=$(git -C "$repo" rev-parse --short=12 origin/main 2>/dev/null) || exit 1
  [ -n "$head_sha" ] || exit 1
  row=$(last_row "$repo" frontier)
  [ -n "$row" ] || exit 0                       # no series yet: bootstrap it
  last_sha=$(printf '%s' "$row" | cut -f4)
  if [ "$last_sha" != "$head_sha" ]; then
    # MAIN MOVING IS NOT THE SAME AS THE FRONTIER MOVING. On a busy night main
    # takes dozens of commits, and most of them cannot change a decode step:
    # this script itself, the results file it writes, docs, other benchmarks.
    # Firing a 25-minute two-box run per commit would make the habit the
    # heaviest consumer of the machines it measures, and would do it while
    # measuring nothing new.
    #
    # So ask whether anything that COMPILES INTO THE BINARY changed. Still one
    # local git call and still no network, which is the contract for a predicate
    # `orient` re-evaluates every 60 s. If the range is not resolvable locally
    # (the last row's sha was never fetched here) this answers DUE, because not
    # knowing is a reason to look rather than a reason to skip.
    if git -C "$repo" cat-file -e "$last_sha^{commit}" 2>/dev/null; then
      changed=$(git -C "$repo" diff --name-only "$last_sha..origin/main" -- \
                  src Cargo.toml Cargo.lock rust-toolchain.toml 2>/dev/null | head -1)
      [ -n "$changed" ] && exit 0                # the binary can have changed
    else
      exit 0
    fi
    # main moved but only outside the binary: fall through to the heartbeat,
    # which still fires eventually and measures box drift at a fixed sha.
  fi
  last_at=$(iso_to_epoch "$(printf '%s' "$row" | cut -f1)")
  [ "${last_at:-0}" -gt 0 ] || exit 0           # unparseable stamp: look at it
  age_h=$(( ( $(date -u +%s) - last_at ) / 3600 ))
  [ "$age_h" -ge "$FRONTIER_HEARTBEAT_H" ] && exit 0   # heartbeat: measure the drift
  exit 1
fi

FORCE=0
if [ "${1:-}" = "--force" ]; then FORCE=1; shift; fi

# ==========================================================================
# THE RUN PHASE -- this half executes ON THE HEAD BOX.
# ==========================================================================
if [ "${1:-}" = "--run" ]; then
  STAGE=${FRONTIER_STAGE:?--run needs FRONTIER_STAGE, the directory this file was staged into}
  HEADNAME=$(hostname)
  # One TAG for the whole run: it names the reservation, the pipe-bench output
  # directory, every log, and the only pattern this script will ever match a
  # process against. A UTC stamp makes it unique to this run by construction.
  TAG=${FRONTIER_LOCK_TAG:-frontier-$(date -u +%Y%m%dT%H%M%SZ)}
  UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  TAILSSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 $FRONTIER_USER@$FRONTIER_TAIL_IP"
  WT=$HOME/$FRONTIER_WT
  PARENT=$HOME/$FRONTIER_PARENT
  BIN=$HOME/$FRONTIER_BINDIR/inkling_forward
  IDS=$HOME/$FRONTIER_IDS
  OUT=/tmp/pipe-$TAG
  RUNLOG=/tmp/frontier-run-$TAG.log
  NOTES=""
  note() { NOTES="${NOTES:+$NOTES;}$1"; }

  say "=== frontier-bench $TAG ==="
  say "  head $HEADNAME   tail $FRONTIER_TAIL_IP"

  # ---- 1. the reservation must be ours ---------------------------------
  # The CONTROL phase takes `scripts/gb10-lock.sh` on both boxes before it
  # starts us, because a box cannot ssh to itself here. What this half can do,
  # and does, is REFUSE unless the reservation is actually ours: a run whose
  # lock was lost or broken must stop, not measure unprotected. Both reads are
  # plain file reads -- local on the head, one `cat` on the tail -- so no
  # pattern crosses a shell and no heredoc can be eaten by an `ssh -n`.
  lock_tag_of() {  # lock_tag_of <ssh-prefix-or-empty>
    local pre=$1
    $pre cat "$HOME/gb10/box.lock.d/info" 2>/dev/null | sed -n 's/^tag=//p'
  }
  if [ -n "$FRONTIER_LOCK_TAG" ]; then
    HTAG=$(lock_tag_of "")
    TTAG=$(lock_tag_of "$TAILSSH")
    if [ "$HTAG" != "$FRONTIER_LOCK_TAG" ] || [ "$TTAG" != "$FRONTIER_LOCK_TAG" ]; then
      say "REFUSING: the gb10 box lock is not ours."
      say "  head holds '${HTAG:-nothing}', tail holds '${TTAG:-nothing}', we are '$FRONTIER_LOCK_TAG'."
      say "  A measurement without a reservation is how two runs OOM-kill each other."
      exit 3
    fi
    say "  reservation: gb10 box lock held on both boxes as '$FRONTIER_LOCK_TAG'"
  else
    say "  !! no FRONTIER_LOCK_TAG -- running WITHOUT a gb10 reservation. Only the idle"
    say "     gate protects this run, and a check is not a reservation."
    note "unreserved"
  fi
  # ---- 2. the idle gate -------------------------------------------------
  # The reservation stops the NEXT agent; the gate catches a run that was
  # already in flight when we reserved. Both, and in that order. It runs before
  # the fetch and before the build, because a cargo build is itself enough load
  # to move another agent's numbers.
  # shellcheck source=lib/box-busy.sh
  . "$STAGE/box-busy.sh" || die "cannot source the staged box-busy.sh"
  gate_both() {
    local out rc
    out=$(box_busy_local); rc=$?
    if [ "$rc" = 0 ]; then say "  HEAD $HEADNAME BUSY:"; printf '%s' "$out" | sed 's/^/      /'; return 1; fi
    out=$(box_busy_remote "$FRONTIER_USER@$FRONTIER_TAIL_IP"); rc=$?
    if [ "$rc" = 0 ]; then say "  TAIL $FRONTIER_TAIL_IP BUSY:"; printf '%s' "$out" | sed 's/^/      /'; return 1; fi
    if [ "$rc" = 2 ]; then say "  TAIL $FRONTIER_TAIL_IP state UNKNOWN -- treating as busy"; return 1; fi
    return 0
  }
  waited=0
  until gate_both; do
    if [ "$waited" -ge "$FRONTIER_WAIT_S" ]; then
      say ""
      say "REFUSING TO MEASURE: a box is busy. Nothing was started, nothing was killed,"
      say "and no row was recorded. This is the NORMAL outcome while another agent is"
      say "measuring. Run again later, or raise FRONTIER_WAIT_S (currently ${FRONTIER_WAIT_S}s)."
      exit 3
    fi
    sleep 30
    waited=$((waited + 30))
  done
  say "  idle gate: both boxes clear"

  # ---- 3. fetch and fast-forward main on both boxes ---------------------
  # FAST-FORWARD ONLY, and it REFUSES rather than resetting. A worktree with
  # tracked edits, or a local `main` that is not an ancestor of origin/main, is
  # somebody doing something, and deciding what is not this script's job.
  ensure_wt() {  # ensure_wt <ssh-prefix-or-empty> <label>
    local pre=$1 label=$2 dirty
    $pre git -C "$PARENT" fetch --quiet origin || { say "  $label: cannot fetch origin"; return 1; }
    if $pre git -C "$PARENT" rev-parse --verify -q main >/dev/null 2>&1; then
      $pre git -C "$PARENT" merge-base --is-ancestor main origin/main || {
        say "  $label: local main is NOT an ancestor of origin/main -- refusing (fast-forward only)."
        say "  $label: it carries commits that are on no remote; push or dispose of them first."
        return 1; }
    fi
    if ! $pre test -e "$WT/.git"; then
      say "  $label: creating the frontier worktree at $WT"
      $pre git -C "$PARENT" worktree prune
      $pre git -C "$PARENT" worktree add -B main "$WT" origin/main || return 1
      return 0
    fi
    dirty=$($pre git -C "$WT" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')
    [ "${dirty:-1}" -eq 0 ] || { say "  $label: $WT has ${dirty:-?} tracked edit(s) -- refusing to touch it"; return 1; }
    $pre git -C "$WT" merge-base --is-ancestor HEAD origin/main || {
      say "  $label: $WT HEAD is not an ancestor of origin/main -- refusing (fast-forward only)"; return 1; }
    $pre git -C "$WT" checkout -q -B main origin/main || return 1
    return 0
  }
  ensure_wt "" "HEAD $HEADNAME" || die "cannot fast-forward main on the head box"
  ensure_wt "$TAILSSH" "TAIL $FRONTIER_TAIL_IP" || die "cannot fast-forward main on the tail box"
  MAIN_SHA=$(git -C "$WT" rev-parse --short=12 HEAD)
  TAIL_MAIN=$($TAILSSH git -C "$WT" rev-parse --short=12 HEAD 2>/dev/null)
  [ -n "$MAIN_SHA" ] && [ "$MAIN_SHA" = "$TAIL_MAIN" ] \
    || die "the two boxes are not on the same main (head $MAIN_SHA, tail ${TAIL_MAIN:-none})"
  say "  main: $MAIN_SHA, fast-forwarded on both boxes"

  # The two floating path-dependencies. Recorded because main's sha does not
  # identify them, and a build against a dirty sibling is not reproducible.
  sib_sha()   { git -C "$HOME/$FRONTIER_SIBDIR/$1" rev-parse --short=12 HEAD 2>/dev/null || echo unknown; }
  sib_dirty() { git -C "$HOME/$FRONTIER_SIBDIR/$1" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' '; }
  TSP_SHA=$(sib_sha triblespace-rs); CUBECL_SHA=$(sib_sha cubecl-graph)
  [ "$(sib_dirty triblespace-rs)" = "0" ] || note "triblespace-rs-dirty"
  [ "$(sib_dirty cubecl-graph)" = "0" ]   || note "cubecl-graph-dirty"
  say "  path-deps: triblespace-rs $TSP_SHA   cubecl-graph $CUBECL_SHA"

  # ---- 4. is there anything to measure ---------------------------------
  PREV=$(last_row "$WT" frontier)
  PREV_SHA=$(printf '%s' "$PREV" | cut -f4)
  KIND=advance
  if [ -n "$PREV" ] && [ "$PREV_SHA" = "$MAIN_SHA" ]; then
    KIND=repeat
    if [ "$FORCE" != 1 ]; then
      last_at=$(iso_to_epoch "$(printf '%s' "$PREV" | cut -f1)")
      age_h=$(( ( $(date -u +%s) - ${last_at:-0} ) / 3600 ))
      if [ "$age_h" -lt "$FRONTIER_HEARTBEAT_H" ]; then
        say ""
        say "NOTHING TO MEASURE: main is unchanged at $MAIN_SHA and the last row is ${age_h}h old"
        say "(the repeat heartbeat is ${FRONTIER_HEARTBEAT_H}h). --force overrides."
        exit 0
      fi
    fi
  fi
  say "  kind: $KIND${PREV_SHA:+   (previous frontier row was at $PREV_SHA)}"

  # ---- 5. build ONCE, keyed by everything that decides the bytes -------
  # main plus both floating path-deps. A repeat at the same key skips the build
  # entirely, which is what makes a nightly heartbeat cost ~18 min not an hour.
  BUILD_KEY="$MAIN_SHA-$TSP_SHA-$CUBECL_SHA"
  BUILDLOG=/tmp/frontier-build-$TAG.log
  mkdir -p "$(dirname "$BIN")"
  if [ "$(cat "$BIN.key" 2>/dev/null)" = "$BUILD_KEY" ] && [ -x "$BIN" ]; then
    say "  build: the staged binary is already keyed $BUILD_KEY -- not rebuilding"
  else
    say "  build: cargo build --release --bin inkling_forward --features $FRONTIER_FEATURES   (key $BUILD_KEY)"
    if ! ( cd "$WT" && PATH="$HOME/.cargo/bin:$PATH" cargo build --release --bin inkling_forward --features "$FRONTIER_FEATURES" ) > "$BUILDLOG" 2>&1; then
      say "  BUILD FAILED -- tail of $BUILDLOG:"
      tail -25 "$BUILDLOG" | sed 's/^/      /'
      die "build failed; nothing measured, nothing recorded"
    fi
    # A zero exit code is not evidence that anything was built -- under
    # `--bins`/`--all-targets` cargo skips unbuildable targets and exits 0, and
    # a future edit to this line is one word away from that form. Demand the
    # artifact itself rather than trusting the status.
    [ -x "$WT/target/release/inkling_forward" ] \
      || die "cargo exited 0 but produced no binary at $WT/target/release/inkling_forward -- check --features $FRONTIER_FEATURES against the [[bin]] required-features in Cargo.toml"
    cp -f "$WT/target/release/inkling_forward" "$BIN" || die "cannot stage the built binary"
    printf '%s\n' "$BUILD_KEY" > "$BIN.key"
    say "  build: ok ($BUILDLOG)"
  fi

  # ---- 6. one binary, both boxes, asserted byte-identical ---------------
  $TAILSSH "mkdir -p $HOME/$FRONTIER_BINDIR" >/dev/null 2>&1
  scp -q -o BatchMode=yes "$BIN" "$FRONTIER_USER@$FRONTIER_TAIL_IP:$BIN" || die "cannot copy the binary to the tail"
  scp -q -o BatchMode=yes "$BIN.key" "$FRONTIER_USER@$FRONTIER_TAIL_IP:$BIN.key" >/dev/null 2>&1
  BIN_SHA=$(sha256sum "$BIN" | awk '{print $1}')
  TBIN_SHA=$($TAILSSH sha256sum "$BIN" 2>/dev/null | awk '{print $1}')
  [ -n "$BIN_SHA" ] && [ "$BIN_SHA" = "$TBIN_SHA" ] \
    || die "the boxes do not hold the same binary (head $BIN_SHA, tail ${TBIN_SHA:-none})"
  say "  binary: $BIN_SHA -- byte-identical on both boxes"

  # The corpus is pinned by CONTENT, not by name: a silently different
  # ctx3732.ids would move the number with nothing in the row to show it.
  IDS_MD5=$(md5sum "$IDS" 2>/dev/null | awk '{print $1}')
  [ "$IDS_MD5" = "$FRONTIER_IDS_MD5" ] \
    || die "$IDS has md5 ${IDS_MD5:-none}, expected $FRONTIER_IDS_MD5 -- refusing to record a number from a different corpus"

  # ---- 7. cool down, re-gate, measure -----------------------------------
  say "  cooldown ${FRONTIER_COOLDOWN_S}s (unified-memory arenas return to the kernel tens of seconds after exit)"
  sleep "$FRONTIER_COOLDOWN_S"
  gate_both || { say "REFUSING: a box went busy during the build. Nothing recorded."; exit 3; }

  ARMS=("frontier:|")
  # SEVERAL experimental arms, split on ';'. This is about RESOLUTION, not
  # convenience. Arms inside one invocation are interleaved by `pipe-bench.sh`,
  # so comparing them is PAIRED and resolves to about the 1.1% this lane
  # reaches; comparing two rows from two SESSIONS is not paired and resolves far
  # worse. Measuring `graph` on Monday and `all-on` on Tuesday therefore answers
  # a strictly weaker question than measuring them side by side, and costs two
  # box sessions to do it. `pipe-bench.sh` already took the arms variadically --
  # only this wrapper was narrowing them to one.
  #
  # ';' because ':' and '|' both already mean something INSIDE an arm
  # ('name:HEADENV|TAILENV'). An env value containing a literal ';' cannot be
  # expressed; nothing in this lane needs one.
  EXTRA_ARMS=()
  if [ -n "$FRONTIER_EXTRA_ARM" ]; then
    OIFS=$IFS; IFS=';'
    for _a in $FRONTIER_EXTRA_ARM; do
      [ -n "$_a" ] && EXTRA_ARMS+=("$_a")
    done
    IFS=$OIFS
    # Guarded: `set -u` is on and an empty-array expansion is an error on the
    # bash the boxes ship.
    [ ${#EXTRA_ARMS[@]} -gt 0 ] && ARMS+=("${EXTRA_ARMS[@]}")
  fi
  say "  arms: ${ARMS[*]}"
  say "  scoreboard lane: reps $FRONTIER_REPS, GEN $FRONTIER_GEN, split $FRONTIER_SPLIT, ctx 3732, INK_KV=1, --overlap, --order fixed"
  HBIN="$BIN" TBIN="$BIN" HPILE="$HOME/$FRONTIER_HPILE" TPILE="$HOME/$FRONTIER_TPILE" \
  TAILHOST="$FRONTIER_TAIL_IP" REMOTE_USER="$FRONTIER_USER" SPLIT="$FRONTIER_SPLIT" \
    timeout --foreground -k 30 "$FRONTIER_RUN_TIMEOUT_S" \
    bash "$WT/scripts/pipe-bench.sh" --overlap --order fixed \
      "$TAG" "$FRONTIER_REPS" "$FRONTIER_GEN" "$IDS" "${ARMS[@]}" > "$RUNLOG" 2>&1
  RUNRC=$?
  if [ "$RUNRC" = 124 ] || [ "$RUNRC" = 137 ]; then
    note "run-timeout-${FRONTIER_RUN_TIMEOUT_S}s"
    say "  !! the run exceeded ${FRONTIER_RUN_TIMEOUT_S}s and was stopped."
    # Reap ONLY this run's own processes, by PID, on both boxes. TAG carries a
    # UTC stamp, so it cannot match another run -- and never a pattern kill.
    for p in $(pgrep -f "$TAG" 2>/dev/null); do [ "$p" = "$$" ] || kill "$p" 2>/dev/null; done
    $TAILSSH "for p in \$(pgrep -f pb_out_$TAG.bin); do [ \"\$p\" = \"\$\$\" ] || kill \"\$p\" 2>/dev/null; done" >/dev/null 2>&1
  fi

  RESULTS=$OUT/results.tsv
  if [ ! -s "$RESULTS" ]; then
    say "  !! no results at $RESULTS. Tail of $RUNLOG:"
    tail -25 "$RUNLOG" | sed 's/^/      /'
    die "the run produced no measurement"
  fi
  # `ungated-after` on its own over-reports, and the first real run proved it.
  # pipe-bench re-gates FIVE SECONDS after the last rep exits, and one of the
  # gate's conditions is a 1-MINUTE load average against a threshold of 2.5 --
  # which our own seven back-to-back reps, each holding ~100 GiB, have just put
  # well above it. The first frontier row tripped this with `util 0% load 3.40
  # compute-apps: none`: no GPU work, no compute apps, no measurement-shaped
  # processes, only the decay tail of the run being measured. So carry the
  # EVIDENCE rather than the verdict, and say when it looks like our own residue
  # rather than a neighbour -- a bare flag makes a clean run and a contaminated
  # one indistinguishable, which is the failure this whole file is about.
  if grep -q 'THE BOXES DID NOT STAY IDLE' "$RUNLOG"; then
    gate_ev=$(sed -n '/gate, after the run/,$p' "$RUNLOG" \
      | sed -n 's/.*util \([0-9]*\)%  load \([0-9.]*\)  compute-apps: \(.*\)/util=\1%,load=\2,apps=\3/p' | head -1)
    gate_procs=$(sed -n '/gate, after the run/,$p' "$RUNLOG" | grep -c 'measurement-shaped processes')
    case "$gate_ev" in
      util=0%*apps=none) [ "${gate_procs:-0}" -eq 0 ] && gate_ev="$gate_ev,likely-own-decay" ;;
    esac
    note "ungated-after${gate_ev:+($gate_ev)}"
  fi
  grep -q 'IDENTITY FAILS' "$RUNLOG" && note "identity-fails"
  grep -q 'may be paging' "$RUNLOG" && note "paging-suspected"
  fails=$(grep -c -E 'HEAD FAILED|TAIL FAILED' "$RUNLOG")
  [ "${fails:-0}" -gt 0 ] && note "rep-failures=$fails"

  # ---- 8. the row ------------------------------------------------------
  med()    { sort -n | awk '{v[NR]=$1} END{if(NR==0){print "-";exit} if(NR%2)printf "%.4f\n",v[(NR+1)/2]; else printf "%.4f\n",(v[NR/2]+v[NR/2+1])/2}'; }
  spread() { sort -n | awk '{v[NR]=$1} END{if(NR==0){print "-";exit} m=(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2; printf "%.2f\n",100.0*(v[NR]-v[1])/m}'; }
  CONFIG="per-decode-step/2node/split$FRONTIER_SPLIT/ctx3732/GEN$FRONTIER_GEN/INK_KV=1/overlap/order-fixed/median-of-n-process-reps/features=$FRONTIER_FEATURES"

  emit_row() {  # emit_row <arm> <kind> <env>
    local arm=$1 kind=$2 envs=$3
    local n tok ms sp prev prev_tok prev_n nmin dr rest rownotes why
    local delta res verdict
    n=$(awk -F'\t' -v a="$arm" 'NR>1 && $1==a' "$RESULTS" | wc -l | tr -d ' ')
    if [ "${n:-0}" -eq 0 ]; then say "  !! arm $arm produced no reps -- not recorded"; return 1; fi
    tok=$(awk -F'\t' -v a="$arm" 'NR>1 && $1==a {print $3}' "$RESULTS" | med)
    ms=$( awk -F'\t' -v a="$arm" 'NR>1 && $1==a {print $4}' "$RESULTS" | med)
    sp=$( awk -F'\t' -v a="$arm" 'NR>1 && $1==a {print $3}' "$RESULTS" | spread)
    rownotes=$NOTES
    if [ "$n" -lt "$FRONTIER_REPS" ]; then
      # A silently short run must NEVER report as though the short count was
      # asked for. Say the true n, and say what the log knows about why.
      why=$(grep -m1 -E 'HEAD FAILED|TAIL FAILED' "$RUNLOG" | tr -d '\t' | cut -c1-60)
      rownotes="${rownotes:+$rownotes;}short-run:n=$n-of-$FRONTIER_REPS${why:+:$why}"
      say "  !! arm $arm completed $n of $FRONTIER_REPS reps. ${why:-No failure line in the log; read $RUNLOG.}"
    fi
    prev=$(last_row "$WT" "$arm")
    delta=-; res=-; verdict=first
    if [ -n "$prev" ]; then
      prev_tok=$(printf '%s' "$prev" | cut -f10)
      prev_n=$(printf '%s' "$prev" | cut -f8)
      nmin=$n; [ "${prev_n:-0}" -lt "$nmin" ] 2>/dev/null && nmin=$prev_n
      # A ROW-TO-ROW COMPARISON IS NOT PAIRED. The 1.1%-at-n=7 figure is the
      # resolution of a PAIRED arm delta inside ONE interleaved run; two medians
      # from two different runs resolve less well, and the wider of the two is
      # the honest threshold. Both are computed; the wider one decides.
      dr=$(awk -v a="$prev_tok" -v b="$tok" -v n="$nmin" -v psd="$PAIRED_SD" -v usd="$UNPAIRED_SD" 'BEGIN{
             if (a+0 <= 0 || n+0 <= 0) { print "- - first"; exit }
             d = 100.0*(b-a)/a;
             rp = 2*psd/sqrt(n);
             ru = 2*usd*sqrt(2)/sqrt(n);
             r  = (ru>rp) ? ru : rp;
             v  = (d>r) ? "better" : ((d < -r) ? "worse" : "inside-resolution");
             printf "%+.2f %.2f %s\n", d, r, v }')
      delta=${dr%% *}; rest=${dr#* }; res=${rest%% *}; verdict=${rest##* }
    fi
    # Accumulated, NOT written to the file here. The commit is rebuilt from
    # scratch on top of whatever origin/main is at push time (see push_row), so
    # the row has to survive a `reset --hard` and be re-appended.
    ROWS="$ROWS$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "$UTC" "$kind" "$arm" "$MAIN_SHA" "$TSP_SHA" "$CUBECL_SHA" "$BIN_SHA" \
      "$n" "$FRONTIER_REPS" "$tok" "$ms" "$sp" "$delta" "$res" "$verdict" \
      "${envs:--}" "$CONFIG" "${rownotes:--}")
"
    ROW_TOK=$tok; ROW_MS=$ms; ROW_N=$n; ROW_SP=$sp
    ROW_DELTA=$delta; ROW_RES=$res; ROW_VERDICT=$verdict
    return 0
  }

  ROWS=""
  emit_row frontier "$KIND" "-" || die "the frontier arm recorded nothing"
  F_TOK=$ROW_TOK; F_MS=$ROW_MS; F_N=$ROW_N; F_SP=$ROW_SP
  F_DELTA=$ROW_DELTA; F_RES=$ROW_RES; F_VERDICT=$ROW_VERDICT
  if [ ${#EXTRA_ARMS[@]} -gt 0 ]; then
    for _a in "${EXTRA_ARMS[@]}"; do
      # `|| true` per arm, deliberately: one experimental arm failing to record
      # must not cost us the frontier row or the other arms. A lane that has
      # never run on this config (INK_GRAPH_LANE on two nodes, as of this
      # commit) is exactly the kind that may not record.
      emit_row "${_a%%:*}" experimental "${_a#*:}" || true
    done
  fi

  # ---- 9. commit and push ----------------------------------------------
  # Built on top of whatever origin/main is AT PUSH TIME, from the frontier
  # worktree, so the commit carries this row and nothing else -- never another
  # agent's unpushed work. The worktree shares the parent repository's hooks, so
  # the box's pre-push leak guard audits it. No --no-verify, ever.
  #
  # THE PUSH MUST SURVIVE main MOVING UNDER IT, and the first real run proved
  # that it did not. Between fetching main and pushing the row there is a build
  # plus seven process-reps -- 20 minutes on 2026-08-27 -- and on an active
  # night main moves inside that window essentially always. It did: the row was
  # committed, the push was rejected `(fetch first)`, and it sat unpushed on the
  # box. For an UNATTENDED benchmark that is the whole failure: the measurement
  # happened, the boxes were held, and the series did not gain a row.
  #
  # The retry does NOT rebase the commit. It rebuilds it: fetch, hard-reset to
  # origin/main, re-append the row, commit, push. A rebase of an append-only
  # file conflicts the moment two runs append at EOF, and the conflict
  # resolution is not interesting -- both rows belong, in either order. Rebuild
  # is conflict-free by construction and always appends onto the newest file,
  # which is why emit_row accumulates into $ROWS rather than writing the file.
  push_row() {
    local attempt
    for attempt in 1 2 3; do
      git -C "$WT" fetch --quiet origin || { say "  !! cannot fetch origin"; return 1; }
      git -C "$WT" reset -q --hard origin/main || return 1
      if [ ! -f "$WT/$RESULTS_REL" ]; then
        mkdir -p "$WT/$(dirname "$RESULTS_REL")"
        printf 'utc\tkind\tarm\tmain_sha\ttriblespace_sha\tcubecl_graph_sha\tbin_sha256\tn\treps_req\ttok_s_med\tms_step_med\tspread_pct\tdelta_pct\tres_pct\tverdict\tenv\tconfig\tnotes\n' \
          > "$WT/$RESULTS_REL"
      fi
      printf '%s' "$ROWS" >> "$WT/$RESULTS_REL"
      git -C "$WT" add "$RESULTS_REL"
      git -C "$WT" commit -q -m "bench: frontier at $MAIN_SHA -- $F_TOK tok/s, $F_MS ms/step, n=$F_N ($KIND)

The scoreboard lane, run unattended by scripts/frontier-bench.sh. The figure is
PER DECODE STEP of the 42-layer two-node pipeline at ctx 3732, INK_GEN=$FRONTIER_GEN,
INK_KV=1, split $FRONTIER_SPLIT, --overlap, --order fixed; median over $F_N process-reps of
$FRONTIER_REPS requested. The frontier arm sets no INK_* switch of its own, so this is
main out of the box and not a cherry-picked arm. Binary $BIN_SHA,
byte-identical on both boxes; floating path-deps triblespace-rs $TSP_SHA and
cubecl-graph $CUBECL_SHA, which main's sha does not pin." || {
        say "  !! nothing to commit -- was a row produced?"; return 1; }
      if [ "$FRONTIER_PUSH" != 1 ]; then
        say "  row committed in $WT (FRONTIER_PUSH=0, not pushing)"; return 0
      fi
      if git -C "$WT" push -q origin HEAD:main 2>"/tmp/frontier-push-$TAG.log"; then
        say "  pushed $(git -C "$WT" rev-parse --short=12 HEAD) to origin/main"
        return 0
      fi
      # Distinguish a RACE from a REFUSAL. A rejected ref means main moved and
      # retrying is right; anything else -- a leak-guard refusal, no network, no
      # credentials -- will fail identically three times, so stop and say so.
      if ! grep -qE 'rejected|non-fast-forward|fetch first' "/tmp/frontier-push-$TAG.log"; then
        say "  !! push REFUSED (not a race). The row is committed in $WT. Reason:"
        sed 's/^/      /' "/tmp/frontier-push-$TAG.log"
        return 1
      fi
      say "  push rejected on attempt $attempt: main moved during the run. Rebuilding the row on top of it."
    done
    say "  !! push still rejected after 3 attempts. The row is committed in $WT and needs a human."
    return 1
  }
  push_row || true

  # ---- 10. the verdict --------------------------------------------------
  say ""
  say "=== FRONTIER $UTC   main $MAIN_SHA   [$KIND] ==="
  say "  $F_TOK tok/s median, $F_MS ms/step median, spread $F_SP%, n=$F_N of $FRONTIER_REPS requested"
  say "  PER decode step of the 42-layer two-node pipeline; AT ctx 3732, GEN $FRONTIER_GEN, INK_KV=1,"
  say "  split $FRONTIER_SPLIT, --overlap, --order fixed, no INK_* switch set (main out of the box)."
  if [ "$F_VERDICT" = first ]; then
    say "  VERDICT: FIRST ROW -- there is nothing to compare against yet."
  else
    say "  VERDICT: $F_VERDICT   ($F_DELTA% against the previous frontier row.)"
    say "           This comparison resolves $F_RES% at 2 sem and it is UNPAIRED -- two medians"
    say "           from two separate runs, not an interleaved A/B -- so a difference smaller"
    say "           than $F_RES% is NOT movement and must not be read as one."
  fi
  [ -n "$NOTES" ] && say "  notes: $NOTES"
  say "  row: $WT/$RESULTS_REL   reps: $RESULTS   run log: $RUNLOG"
  exit 0
fi

# ==========================================================================
# THE CONTROL PHASE -- stage this file and box-busy.sh on the head box and
# re-enter there. Every parameter crosses as an environment ASSIGNMENT, never
# as shell text: a pattern that crosses a shell is the bug this tree has now
# written six times (see scripts/lib/box-busy.sh, failure 2).
# ==========================================================================
REPO=$(find_repo) || die "cannot find the mary checkout (set FRONTIER_REPO)"
SELF=$REPO/scripts/frontier-bench.sh
[ -f "$SELF" ] || die "$SELF is missing"
[ -f "$REPO/scripts/lib/box-busy.sh" ] || die "$REPO/scripts/lib/box-busy.sh is missing"

if [ "$FORCE" != 1 ]; then
  if ! bash "$SELF" --due; then
    say "Nothing to measure: main has not moved since the last frontier row and the"
    say "${FRONTIER_HEARTBEAT_H}h repeat heartbeat has not elapsed. --force overrides."
    exit 0
  fi
fi

# --- the reservation, before anything is started ---------------------------
# `scripts/gb10-lock.sh` is the SHARED box lock every agent takes; see BOX
# SAFETY 2. It is taken from here rather than from the head box because a box
# cannot ssh to itself in this setup. Taken on BOTH boxes or on neither: a
# two-node run that reserves one box has reserved nothing.
LOCK=$REPO/scripts/gb10-lock.sh
[ -f "$LOCK" ] || die "$LOCK is missing -- refusing to run without the shared box lock"
FRONTIER_LOCK_TAG=${FRONTIER_LOCK_TAG:-frontier-$(date -u +%Y%m%dT%H%M%SZ)}
HELD_H=0; HELD_T=0; BEATPID=""
release_all() {
  [ -n "$BEATPID" ] && kill "$BEATPID" 2>/dev/null   # our own child, by PID, never a pattern
  [ "$HELD_H" = 1 ] && GB10_LOCK_TIMEOUT_S=$GB10_LOCK_TIMEOUT_S bash "$LOCK" release "$FRONTIER_HEAD" "$FRONTIER_LOCK_TAG" >/dev/null 2>&1
  [ "$HELD_T" = 1 ] && GB10_LOCK_TIMEOUT_S=$GB10_LOCK_TIMEOUT_S bash "$LOCK" release "$FRONTIER_TAIL" "$FRONTIER_LOCK_TAG" >/dev/null 2>&1
  HELD_H=0; HELD_T=0; BEATPID=""
}
trap release_all EXIT INT TERM

# WAIT AND RETRY, DO NOT GIVE UP ON THE FIRST REFUSAL. There is no queue behind
# gb10-lock.sh: if we walk away when a box is held, nobody hands it back to us
# and the night simply has no frontier row -- which is the one outcome this
# whole script exists to prevent. So poll to a CEILING
# (FRONTIER_LOCK_WAIT_S, default 2 h) and then give up loudly, because a poll
# with no deadline is how this project has produced waiters that spun for days.
#
# BOTH BOXES OR NEITHER, and the head is given back between attempts. Holding
# one box while waiting for the other is how two agents deadlock each other,
# each sitting on half of what the other needs.
#
# Every refusal is logged with the holder's tag and how long they have had it.
# That record is the evidence for whether one shared lock with no queue is
# enough, or whether this eventually needs a real queue.
take_both() {
  local out rc
  out=$(GB10_LOCK_TIMEOUT_S=$GB10_LOCK_TIMEOUT_S bash "$LOCK" take "$FRONTIER_HEAD" "$FRONTIER_LOCK_TAG"); rc=$?
  if [ "$rc" -ne 0 ]; then LOCK_WHY="head $FRONTIER_HEAD: $out"; return 1; fi
  HELD_H=1
  out=$(GB10_LOCK_TIMEOUT_S=$GB10_LOCK_TIMEOUT_S bash "$LOCK" take "$FRONTIER_TAIL" "$FRONTIER_LOCK_TAG"); rc=$?
  if [ "$rc" -ne 0 ]; then
    LOCK_WHY="tail $FRONTIER_TAIL: $out"
    GB10_LOCK_TIMEOUT_S=$GB10_LOCK_TIMEOUT_S bash "$LOCK" release "$FRONTIER_HEAD" "$FRONTIER_LOCK_TAG" >/dev/null 2>&1
    HELD_H=0
    return 1
  fi
  HELD_T=1
  return 0
}
# NEVER BREAK A STALE LOCK ON A BOX THAT IS VISIBLY BUSY. gb10-lock.sh treats
# silence as death, which is the right rule for a mechanism that cannot see the
# holder -- but this caller CAN see it. A holder doing one long uninterrupted
# stretch without calling `refresh` goes stale while still measuring, and taking
# its box would destroy a live reservation. The run phase's own gate would then
# refuse and hand the box straight back, so the outcome is not an OOM; it is a
# reservation deleted and a slot wasted for nothing. Looking first costs two ssh
# round trips per poll and removes the whole case.
#
# It also covers the agent who has not adopted the lock yet: a box busy with an
# UNLOCKED run is not one to reserve either. We wait instead, bounded by the
# same ceiling.
# shellcheck source=lib/box-busy.sh
. "$REPO/scripts/lib/box-busy.sh" || die "cannot source scripts/lib/box-busy.sh"
boxes_look_idle() {
  local rc
  box_busy_remote "$FRONTIER_HEAD" >/dev/null 2>&1; rc=$?
  if [ "$rc" != 1 ]; then LOCK_WHY="head $FRONTIER_HEAD is busy or unreachable (a check that cannot fail is not a check, so this fails closed)"; return 1; fi
  box_busy_remote "$FRONTIER_TAIL" >/dev/null 2>&1; rc=$?
  if [ "$rc" != 1 ]; then LOCK_WHY="tail $FRONTIER_TAIL is busy or unreachable (fails closed)"; return 1; fi
  return 0
}

LOCK_WHY=""
lock_waited=0
until boxes_look_idle && take_both; do
  if [ "$lock_waited" -ge "$FRONTIER_LOCK_WAIT_S" ]; then
    say ""
    say "NEVER GOT A SLOT. The boxes stayed reserved for the whole ${FRONTIER_LOCK_WAIT_S}s ceiling."
    say "  last refusal: $LOCK_WHY"
    say "No row was recorded, and nothing was killed or overridden. A lock you can be"
    say "talked out of is not a lock; report the wait, do not defeat it."
    exit 3
  fi
  say "[$(date -u +%H:%M:%SZ)] reserved -- $LOCK_WHY  (waited ${lock_waited}s of ${FRONTIER_LOCK_WAIT_S}s, retrying in ${FRONTIER_LOCK_POLL_S}s)"
  sleep "$FRONTIER_LOCK_POLL_S"
  lock_waited=$((lock_waited + FRONTIER_LOCK_POLL_S))
done
WAITED_NOTE=""
[ "$lock_waited" -gt 0 ] && WAITED_NOTE=" (after ${lock_waited}s of waiting)"
say "gb10 box lock held on $FRONTIER_HEAD and $FRONTIER_TAIL as '$FRONTIER_LOCK_TAG'$WAITED_NOTE"

# Staleness is SILENCE, so a long holder has to keep speaking. Every
# FRONTIER_REFRESH_S until we are done; killed by PID in release_all.
( while :; do
    sleep "$FRONTIER_REFRESH_S"
    GB10_LOCK_TIMEOUT_S=$GB10_LOCK_TIMEOUT_S bash "$LOCK" refresh "$FRONTIER_HEAD" "$FRONTIER_LOCK_TAG" >/dev/null 2>&1
    GB10_LOCK_TIMEOUT_S=$GB10_LOCK_TIMEOUT_S bash "$LOCK" refresh "$FRONTIER_TAIL" "$FRONTIER_LOCK_TAG" >/dev/null 2>&1
  done ) &
BEATPID=$!

STAGE=/tmp/.frontier-stage.$$
FORCEARG=""
[ "$FORCE" = 1 ] && FORCEARG=--force
ssh -n -o BatchMode=yes "$FRONTIER_HEAD" "mkdir -p $STAGE" || die "cannot reach $FRONTIER_HEAD"
scp -q -o BatchMode=yes "$SELF" "$REPO/scripts/lib/box-busy.sh" "$FRONTIER_HEAD:$STAGE/" \
  || die "cannot stage on $FRONTIER_HEAD"

ssh -n -o BatchMode=yes "$FRONTIER_HEAD" \
  "FRONTIER_STAGE=$STAGE FRONTIER_TAIL_IP=$FRONTIER_TAIL_IP FRONTIER_USER=$FRONTIER_USER \
   FRONTIER_WT=$FRONTIER_WT FRONTIER_PARENT=$FRONTIER_PARENT FRONTIER_SIBDIR=$FRONTIER_SIBDIR \
   FRONTIER_BINDIR=$FRONTIER_BINDIR FRONTIER_IDS=$FRONTIER_IDS FRONTIER_IDS_MD5=$FRONTIER_IDS_MD5 \
   FRONTIER_HPILE=$FRONTIER_HPILE FRONTIER_TPILE=$FRONTIER_TPILE \
   FRONTIER_REPS=$FRONTIER_REPS FRONTIER_GEN=$FRONTIER_GEN FRONTIER_SPLIT=$FRONTIER_SPLIT \
   FRONTIER_LOCK_TAG=$FRONTIER_LOCK_TAG FRONTIER_COOLDOWN_S=$FRONTIER_COOLDOWN_S \
   FRONTIER_RUN_TIMEOUT_S=$FRONTIER_RUN_TIMEOUT_S FRONTIER_HEARTBEAT_H=$FRONTIER_HEARTBEAT_H \
   FRONTIER_WAIT_S=$FRONTIER_WAIT_S FRONTIER_PUSH=$FRONTIER_PUSH \
   FRONTIER_EXTRA_ARM='$FRONTIER_EXTRA_ARM' \
   bash $STAGE/frontier-bench.sh $FORCEARG --run"
rc=$?

ssh -n -o BatchMode=yes "$FRONTIER_HEAD" "rm -rf $STAGE" >/dev/null 2>&1
# The row was committed and pushed from the box; bring this checkout's view of
# origin/main up to date so `--due` and the next comparison can see it.
git -C "$REPO" fetch --quiet origin 2>/dev/null
exit $rc
