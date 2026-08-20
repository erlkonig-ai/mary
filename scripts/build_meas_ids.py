#!/usr/bin/env python3
"""Build a prefill measurement prompt whose ROUTING is representative.

Every prefill number in this tree -- 1213.8 ms down to 174.3 ms -- was taken on
`/tmp/meas512.ids`, which is a five-token phrase repeated 102 times. Five
distinct tokens is five distinct rows for the router to look at, and a
top-6-of-256 router given five rows lands on about 48 experts a layer. Real
English at the same length lands on about 188.

The difference is not small and it is not in the noise:

    512 tokens, layers 0:8, p50 over 30 warm passes, nsys off
      /tmp/meas512.ids       5 distinct tokens    45-55 experts/layer   175.6 ms
      /tmp/meas512real.ids 167 distinct tokens   178-204 experts/layer  328.4 ms

and the whole 153 ms is in the two grouped MoE kernels; every other kernel in
the pass is within noise of itself, because every other kernel's cost depends on
the token COUNT and not on which tokens they are. So the standing benchmark
understates real 512-token prefill by 1.87x, and it understates it precisely in
the lane that the last week of work has been optimising.

This writes the diverse counterpart from the paired harness's own hand-written
prompts -- English prose, code, and arithmetic, which is the mix the model is
actually asked to prefill.

    build_meas_ids.py golden/paired/items_all.json /tmp/meas512real.ids [512]
"""
import json
import struct
import sys

items = sys.argv[1] if len(sys.argv) > 1 else "golden/paired/items_all.json"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/meas512real.ids"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 512

ids = []
for it in json.load(open(items))["items"]:
    ids.extend(it["ids"])
    if len(ids) >= n:
        break
if len(ids) < n:
    raise SystemExit("%s holds only %d ids, need %d" % (items, len(ids), n))
ids = ids[:n]
with open(out, "wb") as fh:
    fh.write(struct.pack("<%dq" % n, *ids))
print("%s: %d tokens, %d distinct" % (out, n, len(set(ids))))
