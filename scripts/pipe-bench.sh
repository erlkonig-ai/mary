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
#   pipe-bench.sh TAG REPS GEN IDS 'name:HEADENV|TAILENV' ...
set -u
TAG=$1; REPS=$2; GEN=$3; IDS=$4; shift 4
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

die() { printf '\n!! %s\n\n' "$*" >&2; exit 2; }

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
  mine=$($pre pgrep -a -f "'inkling_forward|inkling_membw|nsys|ncu'" 2>/dev/null | grep -v pgrep || true)
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

echo "=== pipe-bench $TAG ==="
echo "  head : $(hostname)  layers 0:$SPLIT  pile $HPILE"
echo "  tail : $TAILHOST    layers $SPLIT:42 pile $TPILE"
echo "  ids  : $IDS  ($(( $(stat -c %s "$IDS") / 8 )) tokens)   INK_GEN=$GEN  reps=$REPS"
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

run_rep() {
  local name=$1 henv=$2 tenv=$3 rep=$4
  local hlog="$OUT/$name.rep$rep.head.log" tlog="/tmp/pb_$TAG.$name.$rep.tail.log"
  printf '  %-10s rep %d ... ' "$name" "$rep"
  ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST \
    "setsid nohup env INK_KV=1 INK_GEN=$GEN INK_LAYERS=$SPLIT:42 INK_PIPE=tail:0.0.0.0:$PORT $tenv \
     $TBIN $TPILE $IDS /tmp/pb_out_$TAG.bin </dev/null > $tlog 2>&1 &"
  local i ok=0
  for i in $(seq 1 900); do
    ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "grep -q 'pipe: listening' $tlog" && { ok=1; break; }
    ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "grep -qiE 'panic|Error|refus' $tlog" && { ok=2; break; }
    sleep 2
  done
  if [ "$ok" != 1 ]; then
    echo "TAIL FAILED TO LISTEN (ok=$ok)"
    ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "tail -20 $tlog"
    return 1
  fi
  local t0 t1
  t0=$(date +%s)
  env INK_KV=1 INK_GEN=$GEN INK_LAYERS=0:$SPLIT INK_PIPE=head:$TAILHOST:$PORT $henv \
    "$HBIN" "$HPILE" "$IDS" "$OUT/$name.rep$rep.out.bin" > "$hlog" 2>&1
  local rc=$?
  t1=$(date +%s)
  sleep 5
  ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST "cat $tlog" > "$OUT/$name.rep$rep.tail.log" 2>/dev/null
  # make sure nothing is left holding the GPU on either end
  ssh -n -o BatchMode=yes $REMOTE_USER@$TAILHOST \
    "for p in \$(pgrep -f 'inkling_forward.*INK_PIPE=tail' ; pgrep -f 'inkling_forward'); do kill \$p 2>/dev/null; done" >/dev/null 2>&1
  if [ $rc -ne 0 ]; then echo "HEAD FAILED rc=$rc"; tail -5 "$hlog"; return 1; fi
  # the head prints the WARM summary for a pipe run; the tail prints its own stage figures
  gawk -v arm="$name" -v rep="$rep" -v out="$OUT/results.tsv" '
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
      printf "%.3f tok/s, %.1f ms/step over %d warm passes; E_warm %.3f (identity %+.2f%%), tokens/pass all %s, mean accepted drafts %s\n",
             t, s, ws, Ewarm, err, (tpp==""?"-":tpp), (acc==""?"-":acc)
      if (err>1||err<-1) printf "    !! IDENTITY FAILS by %.2f%%\n", err
      printf "%s\t%d\t%.4f\t%.3f\t%.4f\t%s\t%s\t%d\t%s\n", arm, rep, t, s, Ewarm, (tpp==""?"-":tpp), (acc==""?"-":acc), ws, ctx >> out
    }' "$hlog"
  printf '    %ds wall  head=%s tail=%s\n' "$((t1-t0))" "$hlog" "$OUT/$name.rep$rep.tail.log"
}

[ -f "$OUT/results.tsv" ] || printf 'arm\trep\ttok_s\tstep_ms\tE_warm\ttpp_all\tmean_acc_drafts\twarm_steps\tctx\n' > "$OUT/results.tsv"
for r in $(seq 1 "$REPS"); do
  echo "--- rep $r of $REPS ---"
  for spec in "${ARMS[@]}"; do
    name=${spec%%:*}; rest=${spec#*:}
    henv=${rest%%|*}; tenv=${rest#*|}
    run_rep "$name" "$henv" "$tenv" "$r"
  done
done

echo
echo "--- gate, after the run ---"
gate || echo "!! THE BOXES DID NOT STAY IDLE -- treat the numbers as UNGATED"
echo
echo "=== $TAG: 42 layers, split $SPLIT, two nodes ==="
gawk -F'\t' 'NR>1{n[$1]++; t[$1,n[$1]]=$3; s[$1,n[$1]]=$4; e[$1,n[$1]]=$5; a[$1]=$7; c[$1]=$9;
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
    for(o=2;o<=q;o++){x=ord[o]; d=100.0*(M[x]-M[b])/M[b];
      printf "  %s vs %s (median tok/s): %+.2f%%%s\n",x,b,d,((d<0?-d:d) < (SP[x]>SP[b]?SP[x]:SP[b]) ? "   <- SMALLER THAN THE SPREAD. Not a result." : "")}
  }' "$OUT/results.tsv"
echo
echo "  logs: $OUT"
