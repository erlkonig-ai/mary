"""Hold N GiB of ANONYMOUS, touched memory for a while, then let go.

Anonymous is the whole point. The first attempt at this stressor streamed a
large file instead, and it did nothing: pages read once and never re-referenced
land on the inactive list and are the first thing reclaimed, so a streaming
reader evicts itself and leaves the pile's hot pages alone. Anonymous pages
cannot be dropped -- only swapped -- so the page cache is what must shrink, and
the pile is re-faulted underneath a running forward.

`oom_score_adj` is maxed so that if anything is killed for this, it is this and
not the run being measured.
"""

import sys
import time

gib = int(sys.argv[1])
secs = int(sys.argv[2])

with open("/proc/self/oom_score_adj", "w") as f:
    f.write("1000")

buf = bytearray(gib << 30)
# Touch one byte per 4 MiB: enough to fault every huge page in.
for i in range(0, len(buf), 1 << 22):
    buf[i] = 1

print("held", gib, "GiB", flush=True)
time.sleep(secs)
