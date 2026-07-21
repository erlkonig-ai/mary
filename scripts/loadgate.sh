#!/bin/bash
# Wait for a quiet machine (1-min loadavg below threshold), then run the
# given command. Prints the loadavg observed at launch so measurements can
# carry their honesty asterisk. Usage: loadgate.sh <max-load> <cmd...>
MAX=$1; shift
while :; do
  L=$(sysctl -n vm.loadavg | awk '{print $2}')
  ok=$(echo "$L < $MAX" | bc -l)
  [ "$ok" = "1" ] && break
  echo "loadgate: loadavg $L >= $MAX, waiting 30s…" >&2
  sleep 30
done
echo "loadgate: launching at loadavg $L" >&2
exec "$@"
