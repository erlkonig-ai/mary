set -u
# Hosts, user and paths are PARAMETERS, not literals: a committed script that
# bakes in someone's home directory is both wrong to reuse and a term this
# repo's push hook refuses for the public channel.
TAIL_USER=${TAIL_USER:-$(id -un)}
TAIL_IP=${TAIL_IP:-10.55.0.2}
PROMPTS=${PROMPTS:-$HOME/refprompts}
RUNNER=${RUNNER:-/tmp/topk_run.sh}
# The owed measurement: the stackp1 sweep, which converts the ~1.03x BOUND on
# the free gate into a result. ~4 minutes, three corpora.
#
# PRE-FLIGHT REFUSES; IT DOES NOT CLEAR. The previous version of this script
# killed every `inkling_forward` it found on both boxes to make room, which is
# a pattern-kill on a shared machine and nearly took out another window's ncu
# run. Ownership is decided STRUCTURALLY, by /proc/<pid>/exe, because that
# cannot match another process no matter what the pattern is.
MINE='wt-mtp-tree/target/release/inkling_forward|/tmp/ink_topk'
foreign() {  # foreign <ssh-prefix-or-empty> -> prints any process that is not mine
  local pre=$1
  $pre bash -c 'for p in $(pgrep -x inkling_forward 2>/dev/null); do
      e=$(readlink -f /proc/$p/exe 2>/dev/null)
      case "$e" in *wt-mtp-tree/target/release/inkling_forward|/tmp/ink_topk) ;; *) echo "$p $e";; esac
    done; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v -e wt-mtp-tree -e ink_topk'
}
for side in "HEAD:" "TAIL:ssh -n -o BatchMode=yes ${TAIL_USER}@${TAIL_IP}"; do
  label=${side%%:*}; pre=${side#*:}
  out=$(foreign "$pre")
  if [ -n "$out" ]; then
    echo "REFUSING: $label is not mine to take --"; echo "$out" | sed 's/^/    /'
    exit 3
  fi
done
echo "pre-flight clear on both boxes"
for c in ctx3732 rustcode count; do
  echo "########## CORPUS $c"
  bash "$RUNNER" "$PROMPTS/$c.ids" s_$c 2>&1 \
    | grep -vE "^  tokens |^\s+\[" | cut -c1-200
done
