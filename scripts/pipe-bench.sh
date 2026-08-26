#!/bin/bash
# Two-node pipeline benchmark driver (head local, tail over ssh to $TAILHOST).
#
# TRACKED 2026-08-25. Until now this lived ONLY as ~/pipe-bench.sh on
# spark2, untracked, while its single-node sibling scripts/bench-decode.sh was in
# the repo. Every two-node number this project has quoted came out of an
# unversioned file on one box. It is here so the gate's history travels with it.
#
# TWO GATE DEFECTS WERE FIXED THE DAY IT WAS TRACKED. Both had the same shape --
# a check that could not fail -- and both are guarded in gate_one() below:
#
# 1. AN UNREACHABLE BOX READ AS IDLE. Every test treated an empty ssh reply as
#    quiet: util="" -> [ 0 -gt 5 ] false; procs="" -> [ -n "" ] false; load="" ->
#    awk saw 0 > 2.5 false. So a box that did not answer gated CLEAN, and any run
#    admitted that way carried a "gated" stamp that meant nothing. Now gate_one
#    refuses unless the box answers a sentinel first.
#
# 2. THE TAIL-SIDE PROCESS CHECK HAD NEVER ONCE FIRED. `pgrep -a -f 'a|b|c'` had
#    its quotes eaten by the LOCAL shell, so ssh handed `pgrep -a -f a|b|c` to the
#    REMOTE shell, which parsed the bars as PIPES. Demonstrated live: the old form
#    returned `usage: nsys [--version] ...` -- the last stage of the accidental
#    pipeline printing its help -- while the fixed form returns the real process
#    list. Visible in any old log: HEAD prints "measurement-shaped processes",
#    TAIL never does.
#
# The lesson both share, and the reason it is written here rather than in a commit
# message: a gate whose failure mode is silent success is worse than no gate,
# because it launders an unmeasured run into a measured-looking one.
# pipe-bench.sh -- an IDLE-GATED, INTERLEAVED two-node decode harness.
#
# The 42-layer configuration cannot be driven by scripts/bench-decode.sh: that
# script runs ONE process, and this lane is two, the tail of which must be
# started, waited for, and reaped on every rep. Everything else here is the
# same discipline: gate both boxes before and after, interleave the arms so
# drift lands on all of them, discard the cold passes, report the median.
#
# HEAD is this box. TAIL is $TAILHOST.
#
#   pipe-bench.sh [MODE FLAGS] TAG REPS GEN IDS 'name:HEADENV|TAILENV' ...
#
#   Flags, all optional and all before TAG:
#     --fast            the SCREENING lane: --overlap, --order abba, and the
#                       fast defaults for REPS/GEN when they are given as `-`.
#                       Everything it writes is stamped FAST and goes to
#                       results-fast.tsv, never results.tsv.
#     --overlap         start the head WITHOUT waiting for the tail to listen.
#     --no-overlap      the old serialised start (the scoreboard default).
#     --order abba      reverse the arm order on even reps, so linear box drift
#                       cancels between the arms instead of landing on the one
#                       that always goes second. `fixed` is the old behaviour.
#     --ctx N           truncate IDS to its first N tokens. For REPRESENTATIVE-
#                       NESS, never for speed -- see below.
#     --no-settle       skip the memory settle between reps.
#     -h                this header.
#
#   REPS and GEN accept `-`, meaning "this mode's default": 7/64 on the
#   scoreboard lane, 2/32 on --fast.
#
# ---------------------------------------------------------------------------
# THE FAST PATH, AND WHAT IT GIVES UP (2026-08-26)
#
# A two-arm seven-rep A/B cost ~36 min of wall clock (157 s per rep, measured
# off the mtimes of /tmp/pipe-ab_{main,graph}_r1..r7). Where a rep went:
#
#   ~35 s  tail startup   (index 16.6 s + weight copy 17.9 s)  ]  SERIALISED,
#   ~35 s  head startup   (index 15.4 s + weight copy 19.3 s)  ]  one then the other
#   ~27 s  head first pass -- kernel JIT and the arena, NOT the prompt
#   ~7.5 s decode, 64 steps            <- the entire measurement
#   ~50 s  gate, poll, sleep, reap, and the settle between reps
#
# THE LARGEST SINGLE WASTE WAS NOT A SHORTCUT ANYBODY CHOSE. The driver started
# the TAIL, waited for it to print `pipe: listening`, and only THEN started the
# head -- so two ~35 s startups that contend for nothing (different boxes, an
# idle wire) ran back to back. The tail's own log has been saying so all along:
#
#     pipe: head connected from 10.55.0.2:34300 in 36.4s
#
# 36.4 s of a 117 s rep, spent by a tail sitting on a loaded GPU waiting. The
# binary has retried the connect for INK_PIPE_WAIT (180 s) since the pipe was
# written -- `inkling_forward`'s own comment says "the order the two commands
# are started in used to matter and no longer does" -- so overlapping the two
# starts changes NOTHING about what is measured and returns ~36 s per rep.
# That is why --overlap is offered on the scoreboard lane too, and why it is
# not silently ON there: an overlapped tail begins its prefill immediately
# instead of idling 36 s first, and the arm-to-arm spread on this lane is 1.3%,
# which is small enough that "the GPU's clock state at t=0 differs" is not a
# claim to make without measuring it.
#
# WHAT DOES NOT WORK, with the numbers, so nobody buys the shortcut twice:
#
#  * A SHORTER CONTEXT DOES NOT MAKE THE RUN FASTER. It makes it SLOWER. The
#    27 s "prefill" is the first pass paying for kernel compilation and the
#    resident weight upload, and neither cost reads the row count. Step 0's
#    pass_ms against context, from /tmp/pipe-b_ctx*:
#        ctx   16: 36.8 s      ctx 1024: 27.4 s
#        ctx  128: 37.3 s      ctx 2048: 33.5 s
#        ctx  512: 26.6 s      ctx 3732: 26.6 s
#    and the wall clock to the first WARM step runs monotone the WRONG way --
#    60.5, 58.1, 56.3, 55.9, 54.6, 48.0 s for ctx 16 -> 3732. The step itself
#    is flat (94.8 ms at ctx 576, 94.1 at 1088, 94.8 at 3732), so ctx512 is a
#    REPRESENTATIVE context. It is simply not a cheaper one, and --ctx is
#    offered for the first reason and never the second.
#
#  * REPS INSIDE ONE PROCESS ARE NOT REPS. Setup is per process, so folding the
#    reps into one process looks like a 4x, and it needs no binary change at
#    all: a "rep" of a deterministic decode is just more decode steps, which is
#    what INK_GEN already buys. (INK_REPEAT is NOT this. It asserts the cache
#    off and re-runs the whole prompt -- the uncached prefill lane, wrong by
#    5.9x as scripts/bench-decode.sh records.) It still does not work, because
#    the error a rep is bought to average is PER PROCESS and not per step:
#        within one process, sd of the median over 62 warm steps    0.23%
#          (bootstrap over 1197 detrended warm passes, /tmp/pipe-h[123]_spec0)
#        between processes, same arm and box, sd of ms/step         1.2-1.4%
#          (main 1.40%, graph 1.16%, over r1..r7)
#    Seven blocks inside one process draw the 1.3% lottery ONCE, so in-process
#    repetition buys wall clock and no resolution whatever. A long run also
#    RAMPS: three independent 400-step runs drift +2.8%, +3.4% and +5.2% from
#    their first 64-step block to their last, monotone in all three. Reps stay
#    processes; only their SETUP is worth attacking.
#
# WHAT --fast GIVES UP: resolution, and nothing else. Paired arm deltas over
# r1..r7 have sd 1.45%, so at two standard errors of the mean:
#        reps 7  ->  resolves 1.1%   (the scoreboard)
#        reps 3  ->  resolves 1.7%
#        reps 2  ->  resolves 2.1%   (--fast default)
#        reps 1  ->  resolves 2.9%, and drift no longer cancels at all
# and GEN 64 -> 32 moves the within-process term from 0.23% to 0.33%, which is
# nothing beside 1.3%, for 3.5 s a rep. SCREEN with --fast, CONFIRM on the
# scoreboard lane. The two never share a results file.
set -u

die() { printf '\n!! %s\n\n' "$*" >&2; exit 2; }

# ---- mode ----------------------------------------------------------------
MODE=scoreboard
OVERLAP=0
ORDER=fixed
SETTLE=1
CTX=""
while [ $# -gt 0 ]; do
  case ${1:-} in
    --fast)       MODE=fast; OVERLAP=1; ORDER=abba; shift ;;
    --overlap)    OVERLAP=1; shift ;;
    --no-overlap) OVERLAP=0; shift ;;
    --order)      ORDER=${2:-fixed}; shift 2 ;;
    --ctx)        CTX=${2:-}; shift 2 ;;
    --settle)     SETTLE=1; shift ;;
    --no-settle)  SETTLE=0; shift ;;
    -h|--help)    sed -n '1,/^set -u/p' "$0" | sed -e 's/^#\{1,\} \{0,1\}//' -e '/^set -u/d'; exit 0 ;;
    --)           shift; break ;;
    -*)           die "unknown option $1 (flags go BEFORE TAG)" ;;
    *)            break ;;
  esac
done
case $ORDER in abba|fixed) ;; *) die "--order takes abba or fixed, not $ORDER" ;; esac
[ $# -ge 5 ] || die "usage: pipe-bench.sh [--fast] [--overlap] [--order abba] [--ctx N] TAG REPS GEN IDS 'name:HEADENV|TAILENV' ..."

TAG=$1; REPS=$2; GEN=$3; IDS=$4; shift 4
# `-` means "this mode's default", so a caller never has to remember which
# numbers the scoreboard is defined by.
[ "$REPS" = "-" ] && { if [ "$MODE" = fast ]; then REPS=2; else REPS=7; fi; }
[ "$GEN"  = "-" ] && { if [ "$MODE" = fast ]; then GEN=32; else GEN=64; fi; }
ARMS=("$@")
REMOTE_USER=${REMOTE_USER:-$(id -un)}
TAILHOST=${TAILHOST:-10.55.0.1}
PORT=${PORT:-7801}
HPILE=${HPILE:-~/work-inkling-complete.pile}
TPILE=${TPILE:-~/converted/inkling-small-complete.pile}
HBIN=${HBIN:-$HOME/mary/target/release/inkling_forward}
TBIN=${TBIN:-$HOME/mary/target/release/inkling_forward}
SPLIT=${SPLIT:-21}
OUT=/tmp/pipe-$TAG
mkdir -p "$OUT"

# --ctx truncates the token file. i64 little-endian, so N tokens is 8N bytes --
# the same format INK_FORCE_IDS and the prompt files use. For representativeness
# only; see the header for why it is not a speedup.
if [ -n "$CTX" ]; then
  _full=$IDS
  IDS=$OUT/ids_ctx$CTX.ids
  head -c $((CTX * 8)) "$_full" > "$IDS" || die "cannot truncate $_full to $CTX tokens"
  [ "$(stat -c %s "$IDS")" = "$((CTX * 8))" ] || die "$_full has fewer than $CTX tokens"
fi

# A fast number must never be able to land in the file a scoreboard number is
# read from. Two files, not one column -- a column can be dropped by the next
# person's awk, a filename cannot.
if [ "$MODE" = fast ]; then RESULTS="$OUT/results-fast.tsv"; else RESULTS="$OUT/results.tsv"; fi

gate_one() {  # gate_one <label> <ssh-prefix-or-empty>
  local label=$1 pre=$2 util procs load mine
  # REACHABILITY FIRST (added 2026-08-25). Every test below reads an empty reply
  # as quiet: util="" -> [ 0 -gt 5 ] false; procs="" -> [ -n "" ] false; load=""
  # -> awk sees 0 > 2.5 false. So an UNREACHABLE box used to gate CLEAN, and a
  # gate that cannot fail is not a gate. Refuse unless the box answers a sentinel.
  if ! $pre true 2>/dev/null || [ "$($pre echo __UP__ 2>/dev/null)" != "__UP__" ]; then
    echo "  $label: UNREACHABLE — refusing (an unanswered box is not an idle box)"
    return 1
  fi
  util=$($pre nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -dc 0-9)
  procs=$($pre nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  load=$($pre cat /proc/loadavg 2>/dev/null | awk '{print $1}')
  # The pattern MUST survive two shells. Unquoted, the local shell strips the
  # quotes and ssh hands `pgrep -a -f a|b|c` to the REMOTE shell, which parses the
  # bars as PIPES -- so this check had never once fired on the tail side (visible
  # in any log: HEAD prints "measurement-shaped processes", TAIL never does).
  # THE PATTERN MUST SURVIVE EXACTLY AS MANY SHELLS AS IT CROSSES, and the two
  # sides cross a different number. Remote ($pre set) goes through the local shell
  # AND the remote one, so the quotes must be doubled or the remote shell parses
  # the bars as PIPES. Local ($pre empty) crosses ONE shell, so those same doubled
  # quotes arrive as literal characters and pgrep searches for a pattern
  # containing apostrophes, which never matches anything.
  #
  # Both mistakes have now been made here in sequence: the original unquoted form
  # silently disabled the TAIL check, and the 2026-08-25 fix for it silently
  # disabled the HEAD check. Verified on Linux from a script file, so the pattern
  # was not in the invoking command line and could not self-match: with a target
  # process alive, the doubled-quote form matched NOTHING while the bare form
  # matched it. Branch explicitly rather than trying to find one string that works
  # for both -- a single clever string is what produced two unfailable checks.
  local _pat='inkling_forward|inkling_membw|nsys|ncu'
  if [ -z "$pre" ]; then
    mine=$(pgrep -a -f "$_pat" 2>/dev/null | grep -v pgrep || true)
  else
    mine=$($pre "pgrep -a -f '$_pat'" 2>/dev/null | grep -v pgrep || true)
  fi
  echo "  $label: util ${util}%  load $load  compute-apps: ${procs:-none}"
  [ -n "$mine" ] && echo "      measurement-shaped processes: $mine"
  local bad=0
  [ "${util:-0}" -gt 5 ] 2>/dev/null && bad=1
  [ -n "$procs" ] && bad=1
  [ -n "$mine" ] && bad=1
  awk -v l="$load" 'BEGIN{exit !(l+0 > 2.5)}' && bad=1
  return $bad
}

gate() {
  echo "--- idle gate ---"
  gate_one "HEAD $(hostname)" "" || return 1
  gate_one "TAIL $TAILHOST" "ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST" || return 1
  return 0
}

# ---- the memory settle ----------------------------------------------------
#
# A rep holds ~100 GiB of unified memory and the kernel takes TENS OF SECONDS
# to hand it back after the process exits, so a rep that starts too soon does
# not get a slower GPU, it gets page faults -- and they land on the warm steps,
# not the cold ones the discard removes. bench-decode.sh has had this for a
# while; this lane had only `sleep 5`, and every two-node number taken in a
# single invocation of this script was exposed to it.
#
# It is a CONDITION, not a sleep, which is also why it is faster than the
# blind >=60 s it replaces: it usually returns immediately.
#
# Both boxes, because either one paging ruins the pair. `grep MemAvailable
# /proc/meminfo` is sent to the tail with NO shell metacharacters in it and the
# arithmetic is done locally -- the pattern crosses zero shells, which is the
# whole lesson of scripts/lib/box-busy.sh.
MEM_BASE_H=""; MEM_BASE_T=""; SETTLE_LAST=0; SETTLE_TOTAL=0
SETTLE_MAX=${SETTLE_MAX:-120}
mem_h() { grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print int($2/1048576)}'; }
mem_t() { ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print int($2/1048576)}'; }

settle() {
  SETTLE_LAST=0
  [ "$SETTLE" = 1 ] || return 0
  local h t wh wt waited=0
  h=$(mem_h); t=$(mem_t)
  [ -n "${h:-}" ] || return 0            # no /proc/meminfo: a no-op, not a guess
  # An unmeasurable tail must not become a tail that is FOREVER below its
  # baseline -- that would spend SETTLE_MAX on every rep while looking like
  # patience. The gate has already refused an unreachable box; here an empty
  # reply means "do not gate on the tail", not "the tail is full".
  if [ -z "${t:-}" ]; then MEM_BASE_T=0; t=0; fi
  [ -z "$MEM_BASE_H" ] && MEM_BASE_H=$h
  [ -z "$MEM_BASE_T" ] && MEM_BASE_T=${t:-0}
  wh=$(( MEM_BASE_H * 95 / 100 )); wt=$(( MEM_BASE_T * 95 / 100 ))
  while [ $waited -lt "$SETTLE_MAX" ]; do
    if [ "${h:-0}" -ge "$wh" ] && [ "${t:-0}" -ge "$wt" ]; then break; fi
    sleep 5; waited=$((waited + 5)); h=$(mem_h); t=$(mem_t)
  done
  SETTLE_LAST=$waited; SETTLE_TOTAL=$((SETTLE_TOTAL + waited))
  if [ "${h:-0}" -lt "$wh" ] || [ "${t:-0}" -lt "$wt" ]; then
    echo "    !! still ${h}/${MEM_BASE_H} GiB head, ${t}/${MEM_BASE_T} GiB tail after ${waited}s -- this rep may be paging"
  fi
}

# ---- the banner -----------------------------------------------------------
NTOK=$(( $(stat -c %s "$IDS") / 8 ))
if [ "$MODE" = fast ]; then
  STAMP="FAST"
  BANNER="FAST PATH -- SCREENING ONLY, NOT COMPARABLE TO THE SCOREBOARD"
else
  STAMP="scoreboard"
  BANNER="scoreboard lane"
fi
CONFIG="mode=$MODE reps=$REPS gen=$GEN ctx=$NTOK split=$SPLIT kv=1 overlap=$OVERLAP order=$ORDER settle=$SETTLE cold_discard=2"

echo "=== pipe-bench $TAG  [$BANNER] ==="
echo "  CONFIG     : $CONFIG"
echo "  scoreboard : ctx 3732, split 21, reps 7, GEN 64, INK_KV=1, stall-filtered, order fixed, overlap 0"
if [ "$MODE" = fast ]; then
  echo "  !! A FAST NUMBER IS NOT A SCOREBOARD NUMBER. At reps=$REPS the paired"
  echo "     arm delta resolves ~$(awk -v r="$REPS" 'BEGIN{printf "%.1f", 2*1.45/sqrt(r)}')% at two standard errors (sd 1.45% per paired rep,"
  echo "     measured over /tmp/pipe-ab_{main,graph}_r1..r7). Anything smaller is not a result."
  echo "     Results go to $(basename "$RESULTS"), never results.tsv."
fi
echo "  head : $(hostname)  layers 0:$SPLIT  pile $HPILE"
echo "  tail : $TAILHOST    layers $SPLIT:42 pile $TPILE"
echo "  ids  : $IDS  ($NTOK tokens)   INK_GEN=$GEN  reps=$REPS"
echo "  head bin sha256 $(sha256sum "$HBIN" | awk '{print $1}')"
echo "  tail bin sha256 $(ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST sha256sum "$TBIN" | awk '{print $1}')"
for _try in $(seq 1 240); do
  gate && break
  echo "  ... a box is not idle; waiting (attempt $_try). NOT --allow-busy."
  sleep 30
done
gate || die "REFUSING TO MEASURE: a box never went idle in two hours."
echo

scp -q -o BatchMode=yes "$IDS" $REMOTE_USER@$TAILHOST:"$IDS" || die "cannot stage ids on the tail"

# The tail processes this rep may have left behind. Kept out of the ssh command
# line so `pgrep -f inkling_forward` cannot match the shell that is running it
# -- the self-match that scripts/lib/box-busy.sh calls failure 1, and which the
# old inline form had: it computed the kill list with its own PID in it.
reap_tail() {
  # MATCHED ON THIS RUN'S OWN OUTPUT PATH, which no other run on the box can
  # have. Two reasons, and the second is the important one:
  #
  #  * The old form's first pattern, `inkling_forward.*INK_PIPE=tail`, could
  #    never match. `env A=1 prog args` EXECS prog, so the assignments are in
  #    the environment and not in the command line pgrep -f reads. It was the
  #    blanket `pgrep -f inkling_forward` beside it that did all the work --
  #    another check in this file whose failure mode was silence.
  #  * And that blanket pattern kills EVERY inkling_forward on the box it runs
  #    on, including another agent's measurement. These boxes are shared, and
  #    this is not hypothetical: on 2026-08-26 the old form took out reps 3 and
  #    4 of a concurrent 7-rep `pipe-revA` run (rep 3 died mid-layer-sweep at
  #    22:48, rep 4 one second after it connected at 22:49; both are simply
  #    ABSENT from that run's results.tsv, which reported 5 reps as though it
  #    had asked for 5). A reap that cannot tell its own process from a
  #    neighbour's is the same defect as a gate that cannot fail: the damage is
  #    silent and lands on someone else's numbers.
  #
  # `$$` excludes the remote shell, whose own command line contains the marker
  # (failure 1 of scripts/lib/box-busy.sh). pgrep excludes itself.
  ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST \
    "for p in \$(pgrep -f pb_out_$TAG.bin); do [ \"\$p\" = \"\$\$\" ] || kill \"\$p\" 2>/dev/null; done" \
    >/dev/null 2>&1 || true
}

run_rep() {
  local name=$1 henv=$2 tenv=$3 rep=$4
  local hlog="$OUT/$name.rep$rep.head.log" tlog="/tmp/pb_$TAG.$name.$rep.tail.log"
  printf '  %-10s rep %d ... ' "$name" "$rep"
  settle
  local t0 t1 rc hpid i ok
  # t0 BEFORE the tail is launched. The old t0 was taken after the wait for
  # `pipe: listening`, so the "wall" it printed was the head's alone and the
  # tail's ~35 s startup was invisible in a report that looked complete: 74 s
  # printed against a 117 s rep.
  t0=$(date +%s)
  ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST \
    "setsid nohup env INK_KV=1 INK_GEN=$GEN INK_LAYERS=$SPLIT:42 INK_PIPE=tail:0.0.0.0:$PORT $tenv \
     $TBIN $TPILE $IDS /tmp/pb_out_$TAG.bin </dev/null > $tlog 2>&1 &"

  if [ "$OVERLAP" = 1 ]; then
    # Start the head NOW and let pipe_connect retry -- it backs off to 2 s and
    # is bounded by INK_PIPE_WAIT. Bounded low enough here that a tail which
    # never listens costs a minute and a half rather than three.
    env INK_PIPE_WAIT=${INK_PIPE_WAIT:-150} INK_KV=1 INK_GEN=$GEN INK_LAYERS=0:$SPLIT \
      INK_PIPE=head:$TAILHOST:$PORT $henv \
      "$HBIN" "$HPILE" "$IDS" "$OUT/$name.rep$rep.out.bin" > "$hlog" 2>&1 &
    hpid=$!
    # Liveness, at 10 s, only until the head says it is connected -- four ssh
    # round trips over the ~35 s of overlapped startup rather than one every
    # two seconds, and none at all once the pair has met.
    for i in $(seq 1 20); do
      kill -0 "$hpid" 2>/dev/null || break
      grep -q 'pipe: connected to the tail' "$hlog" 2>/dev/null && break
      if ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "grep -qiE 'panic|Error|refus' $tlog" 2>/dev/null; then
        echo "TAIL FAILED TO START"
        ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "tail -20 $tlog"
        kill "$hpid" 2>/dev/null; wait "$hpid" 2>/dev/null
        reap_tail
        return 1
      fi
      sleep 10
    done
    wait "$hpid"; rc=$?
  else
    ok=0
    for i in $(seq 1 900); do
      ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "grep -q 'pipe: listening' $tlog" && { ok=1; break; }
      ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "grep -qiE 'panic|Error|refus' $tlog" && { ok=2; break; }
      sleep 2
    done
    if [ "$ok" != 1 ]; then
      echo "TAIL FAILED TO LISTEN (ok=$ok)"
      ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "tail -20 $tlog"
      reap_tail
      return 1
    fi
    env INK_KV=1 INK_GEN=$GEN INK_LAYERS=0:$SPLIT INK_PIPE=head:$TAILHOST:$PORT $henv \
      "$HBIN" "$HPILE" "$IDS" "$OUT/$name.rep$rep.out.bin" > "$hlog" 2>&1
    rc=$?
  fi
  t1=$(date +%s)
  sleep 5
  ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "cat $tlog" > "$OUT/$name.rep$rep.tail.log" 2>/dev/null
  reap_tail
  if [ $rc -ne 0 ]; then echo "HEAD FAILED rc=$rc"; tail -5 "$hlog"; return 1; fi
  # the head prints the WARM summary for a pipe run; the tail prints its own stage figures
  gawk -v arm="$name" -v rep="$rep" -v out="$RESULTS" -v stamp="$STAMP" '
    /WARM steps only/ { if (match($0, /\(([0-9.]+) ms\/step over ([0-9]+) steps/, m)) { s=m[1]; ws=m[2] } }
    /WARM per TOKEN/  { if (match($0, /\(([0-9.]+) tok\/s over ([0-9]+) tokens/, m)) { t=m[1]; wt=m[2] } }
    /tokens per pass/ { tpp=$NF }
    /draft tokens accepted per verify pass/ { if (match($0, /mean ([0-9.]+)/, m)) acc=m[1] }
    /TOKENS.SEC/ { if (match($0, /: *([0-9.]+)/, m)) pooled=m[1] }
    /pass_ms [0-9]/ { if (match($0, /ctx ([0-9]+)/, c)) ctx=c[1] }
    END {
      if (s=="") { print "    !! no WARM lines in the head log"; exit 3 }
      Ewarm = (ws>0) ? wt/ws : 0
      pred  = Ewarm * 1000.0 / s
      err   = (t>0) ? 100.0*(pred-t)/t : 999
      printf "%.3f tok/s, %.1f ms/step over %d warm passes [%s]; E_warm %.3f (identity %+.2f%%), tokens/pass all %s, mean accepted drafts %s\n",
             t, s, ws, stamp, Ewarm, err, (tpp==""?"-":tpp), (acc==""?"-":acc)
      if (err>1||err<-1) printf "    !! IDENTITY FAILS by %.2f%%\n", err
      printf "%s\t%d\t%.4f\t%.3f\t%.4f\t%s\t%s\t%d\t%s\t%s\n", arm, rep, t, s, Ewarm, (tpp==""?"-":tpp), (acc==""?"-":acc), ws, ctx, stamp >> out
    }' "$hlog"
  printf '    %ds wall (settle %ds)  head=%s tail=%s\n' "$((t1-t0))" "$SETTLE_LAST" "$hlog" "$OUT/$name.rep$rep.tail.log"
}

[ -f "$RESULTS" ] || printf 'arm\trep\ttok_s\tstep_ms\tE_warm\ttpp_all\tmean_acc_drafts\twarm_steps\tctx\tmode\n' > "$RESULTS"
RUN0=$(date +%s)
for r in $(seq 1 "$REPS"); do
  echo "--- rep $r of $REPS ---"
  # --order abba reverses the arm order on even reps. With a fixed order the
  # arm that always goes second carries every bit of drift that accumulates
  # inside a rep; reversing makes a linear drift cancel between the arms
  # exactly. It costs nothing and it is not the scoreboard's behaviour, so it
  # is off unless asked for.
  ORD=("${ARMS[@]}")
  if [ "$ORDER" = "abba" ] && [ $(( r % 2 )) -eq 0 ]; then
    ORD=()
    for (( i=${#ARMS[@]}-1; i>=0; i-- )); do ORD+=("${ARMS[$i]}"); done
  fi
  for spec in "${ORD[@]}"; do
    name=${spec%%:*}; rest=${spec#*:}
    henv=${rest%%|*}; tenv=${rest#*|}
    run_rep "$name" "$henv" "$tenv" "$r"
  done
done
RUN1=$(date +%s)

echo
echo "--- gate, after the run ---"
gate || echo "!! THE BOXES DID NOT STAY IDLE -- treat the numbers as UNGATED"
echo
NPROC=$(( REPS * ${#ARMS[@]} ))
echo "=== $TAG: 42 layers, split $SPLIT, two nodes  [$BANNER] ==="
echo "  CONFIG     : $CONFIG"
printf '  wall       : %ds for %d process-reps = %ds each (settle %ds of it)\n' \
  "$((RUN1-RUN0))" "$NPROC" "$(( (RUN1-RUN0) / (NPROC>0?NPROC:1) ))" "$SETTLE_TOTAL"
gawk -F'\t' -v banner="$BANNER" -v reps="$REPS" 'NR>1{n[$1]++; t[$1,n[$1]]=$3; s[$1,n[$1]]=$4; e[$1,n[$1]]=$5; a[$1]=$7; c[$1]=$9;
    if(!($1 in seen)){seen[$1]=1; ord[++q]=$1}}
  function med(v,k,  i,w,j){for(i=1;i<=k;i++)w[i]=v[i]; asort(w); return (k%2)?w[int(k/2)+1]:(w[int(k/2)]+w[int(k/2)+1])/2}
  END{ printf "  %-10s %4s %13s %14s %9s %9s %8s\n","arm","reps","MEDIAN tok/s","MEDIAN ms/step","E_warm","acc_drafts","spread";
    for(o=1;o<=q;o++){x=ord[o]; k=n[x];
      for(i=1;i<=k;i++){tv[i]=t[x,i]; sv[i]=s[x,i]; ev[i]=e[x,i]}
      mt=med(tv,k); ms=med(sv,k); me=med(ev,k);
      lo=tv[1];hi=tv[1]; for(i=1;i<=k;i++){if(tv[i]<lo)lo=tv[i]; if(tv[i]>hi)hi=tv[i]}
      sp=100.0*(hi-lo)/mt; M[x]=mt; SP[x]=sp;
      printf "  %-10s %4d %13.3f %14.1f %9.3f %9s %7.1f%%\n",x,k,mt,ms,me,a[x],sp}
    print ""; b=ord[1];
    # The resolution this many paired reps buys, from the 1.45% paired sd
    # measured over /tmp/pipe-ab_{main,graph}_r1..r7. Printed beside the delta
    # so a difference is never read without the size of difference this run
    # was able to see.
    res = 2*1.45/sqrt(reps);
    for(o=2;o<=q;o++){x=ord[o]; d=100.0*(M[x]-M[b])/M[b];
      printf "  %s vs %s (median tok/s): %+.2f%%   [%s, reps=%d resolves ~%.1f%% at 2 sem]%s\n",x,b,d,banner,reps,res,
             ((d<0?-d:d) < res ? "   <- SMALLER THAN THIS RUN CAN RESOLVE. Not a result." : "")}
  }' "$RESULTS"
echo
echo "  logs: $OUT   results: $RESULTS"
