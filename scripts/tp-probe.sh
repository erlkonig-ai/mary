#!/bin/bash
# tp-probe.sh -- run tp_allreduce_probe on BOTH boxes and say which wire it used.
#
# The numbers this collects are only worth having if two questions are answered
# in the same breath, and both have burned this project before:
#
#   WHICH BINARY.    A stale worktree on one end nearly derailed two
#                    investigations this week. So the commit AND the sha256 of
#                    the binary on EACH end are printed before anything runs,
#                    and a mismatch is refused rather than noted.
#   WHICH TRANSPORT. NCCL will fall back from RDMA to sockets silently, and on
#                    these boxes that is ~3.68 us against ~185 us -- a factor of
#                    25, which does not adjust the verdict, it INVERTS it. The
#                    only witness is NCCL's own banner, so this always runs with
#                    NCCL_DEBUG=INFO and greps the result for NET/IB against
#                    NET/Socket. A run whose transport cannot be established
#                    prints a refusal in place of a conclusion.
#
# Same discipline as pipe-bench.sh otherwise: gate both boxes on a sentinel
# first (an unreachable box is NOT an idle box), never --allow-busy, and keep
# the full per-rep series rather than a summary.
#
#   tp-probe.sh [TAG] [REPS] [COLLECTIVES_PER_TOKEN]
set -u
TAG=${1:-tp}
REPS=${2:-9}
NCOLL=${3:-86}
REMOTE_USER=${REMOTE_USER:-$(id -un)}
# The PEER on the fast fabric. This is the ConnectX pair, not the management
# NIC and not ZeroTier: measuring the wrong NIC is the exact mistake
# interconnect_probe.sh was written to stop.
PEER=${PEER:-10.55.0.2}
SSHHOST=${SSHHOST:-spark2-zt}
PORT=${PORT:-7899}
RBIN=${RBIN:-$HOME/mary/target/release/tp_allreduce_probe}
LBIN=${LBIN:-$HOME/mary/target/release/tp_allreduce_probe}
REPO=${REPO:-$HOME/mary}
RREPO=${RREPO:-$HOME/mary}
OUT=/tmp/tp-$TAG
mkdir -p "$OUT"

SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8 $REMOTE_USER@$SSHHOST"

die() { printf '\n!! %s\n\n' "$*" >&2; exit 2; }

# --- which interface actually reaches the peer -----------------------------
IFACE=${IFACE:-$(ip -o route get "$PEER" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)}
[ -n "$IFACE" ] || die "no route to $PEER from here; set IFACE= and PEER="
HCA=${HCA:-$(ls -1 "/sys/class/net/$IFACE/device/infiniband" 2>/dev/null | head -1)}
# The peer's own name for the fast interface, so its NCCL_SOCKET_IFNAME is its
# own and not a copy of ours -- the two boxes do not name the ConnectX port
# identically (enp1s0f0np0 here, and it is not safe to assume there).
RIFACE=${RIFACE:-$($SSH "ip -o -4 addr show | awk '/10\.55\./ {print \$2; exit}'")}

echo "=== tp-probe $TAG ==="
echo "  rank 0    $(hostname)   iface $IFACE   hca ${HCA:-none}"
echo "  rank 1    $SSHHOST via $PEER            iface ${RIFACE:-?}"
echo "  reps $REPS   collectives/token $NCOLL   rendezvous port $PORT"

# --- provenance: same commit, and say the sha256 of both binaries -----------
LCOMMIT=$(git -C "$REPO" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
RCOMMIT=$($SSH "git -C $RREPO rev-parse --short=12 HEAD" 2>/dev/null || echo unknown)
LDIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
RDIRTY=$($SSH "git -C $RREPO status --porcelain | wc -l" 2>/dev/null | tr -d ' ')
LSHA=$(sha256sum "$LBIN" 2>/dev/null | awk '{print $1}')
RSHA=$($SSH "sha256sum $RBIN" 2>/dev/null | awk '{print $1}')
echo "  rank 0 commit $LCOMMIT (${LDIRTY:-?} dirty)  bin ${LSHA:-MISSING}"
echo "  rank 1 commit $RCOMMIT (${RDIRTY:-?} dirty)  bin ${RSHA:-MISSING}"
[ -n "$LSHA" ] || die "no binary at $LBIN on this box"
[ -n "$RSHA" ] || die "no binary at $RBIN on $SSHHOST"
[ "$LCOMMIT" = "$RCOMMIT" ] || die "the two ends are on DIFFERENT commits ($LCOMMIT vs $RCOMMIT). \
A two-node figure taken across two trees measures neither of them."
if [ "$LSHA" != "$RSHA" ]; then
  echo "  !! the two binaries differ by sha256. That is legitimate only if the boxes"
  echo "     genuinely built different bytes from the same commit; say so in the report."
fi

# --- idle gate, sentinel first ---------------------------------------------
gate_one() {  # gate_one <label> <ssh-prefix-or-empty>
  local label=$1 pre=$2 util procs load
  if ! $pre true 2>/dev/null || [ "$($pre echo __UP__ 2>/dev/null)" != "__UP__" ]; then
    echo "  $label: UNREACHABLE — refusing (an unanswered box is not an idle box)"
    return 1
  fi
  util=$($pre nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -dc 0-9)
  procs=$($pre nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  load=$($pre cat /proc/loadavg 2>/dev/null | awk '{print $1}')
  echo "  $label: util ${util:-?}%  load ${load:-?}  compute-apps: ${procs:-none}"
  local bad=0
  [ "${util:-0}" -gt 5 ] 2>/dev/null && bad=1
  [ -n "$procs" ] && bad=1
  awk -v l="${load:-0}" 'BEGIN{exit !(l+0 > 2.5)}' && bad=1
  return $bad
}
echo "--- idle gate ---"
for _try in $(seq 1 240); do
  gate_one "RANK0 $(hostname)" "" && gate_one "RANK1 $SSHHOST" "$SSH" && break
  echo "  ... a box is not idle; waiting (attempt $_try). NOT --allow-busy."
  sleep 30
done
gate_one "RANK0 $(hostname)" "" && gate_one "RANK1 $SSHHOST" "$SSH" \
  || die "REFUSING TO MEASURE: a box never went idle in two hours."
echo

# --- run, with the transport banner captured on both ends ------------------
#
# NCCL_DEBUG=INFO on BOTH ranks, because the banner is per rank and a fallback
# can be one-sided: one end on IB and the other on sockets still costs the
# socket latency, and only the socket end says so.
#
# Rank 0 starts first because it BINDS the rendezvous. Rank 1 retries for three
# minutes, so the order is not load-bearing; it just keeps connection-refused
# noise out of the log.
RLOG=/tmp/tp_probe_$TAG.rank1.log
LLOG=$OUT/rank0.log
COMMON="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
LENV="$COMMON NCCL_SOCKET_IFNAME=$IFACE"
RENV="$COMMON NCCL_SOCKET_IFNAME=${RIFACE:-$IFACE}"
[ -n "$HCA" ] && LENV="$LENV NCCL_IB_HCA=$HCA"

echo "--- run ---"
echo "  rank 0 env: $LENV"
echo "  rank 1 env: $RENV"
env $LENV INK_TP=0:2 "$LBIN" "0.0.0.0:$PORT" "$REPS" "$NCOLL" > "$LLOG" 2>&1 &
L0PID=$!
sleep 1
$SSH "setsid nohup env $RENV INK_TP=1:2 $RBIN $PEER:$PORT $REPS $NCOLL </dev/null > $RLOG 2>&1" &
R1PID=$!
wait $L0PID; LRC=$?
wait $R1PID 2>/dev/null || true
$SSH "cat $RLOG" > "$OUT/rank1.log" 2>/dev/null

# --- the transport verdict, which outranks the numbers ---------------------
verdict() {  # verdict <label> <logfile>
  local label=$1 f=$2 ib sock
  ib=$(grep -c "NET/IB" "$f" 2>/dev/null || echo 0)
  sock=$(grep -c "NET/Socket" "$f" 2>/dev/null || echo 0)
  if [ "$ib" -gt 0 ] && [ "$sock" -eq 0 ]; then
    echo "  $label: NET/IB  ($ib lines) — RDMA, the path the costing assumes"
  elif [ "$sock" -gt 0 ] && [ "$ib" -eq 0 ]; then
    echo "  $label: NET/Socket ($sock lines) — !! KERNEL SOCKETS. ~25x the latency."
    echo "        Every collective figure from this run belongs to the wrong wire."
  elif [ "$ib" -gt 0 ] && [ "$sock" -gt 0 ]; then
    echo "  $label: MIXED — NET/IB $ib, NET/Socket $sock. Channels differ; read the banner."
  else
    echo "  $label: transport UNKNOWN — no NET/ line. Was NCCL_DEBUG=INFO set?"
  fi
  grep -E "NET/(IB|Socket)" "$f" 2>/dev/null | head -4 | sed 's/^/        /'
}
echo
echo "--- transport ---"
verdict "rank 0" "$LLOG"
verdict "rank 1" "$OUT/rank1.log"

echo
echo "--- rank 0 numbers ---"
grep -vE "NCCL INFO|^$" "$LLOG" | tail -40
echo
echo "  logs: $LLOG  $OUT/rank1.log"
[ $LRC -eq 0 ] || die "rank 0 exited $LRC — see $LLOG"
