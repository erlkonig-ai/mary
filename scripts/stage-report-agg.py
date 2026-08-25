#!/usr/bin/env python3
"""Fold a `bench-decode.sh` run's per-pass stage reports into one table.

`bench-decode.sh` reports ms/step per arm. `inkling_forward` prints a whole
stage report per PASS -- host buckets, the router's three-way split, the
`INK_STAGE_SYNC=1` device-per-stage block, the outer brackets. A single pass's
report is one sample of a distribution whose spread is several ms, so reading
one by eye is how a 15.6 ms bucket gets quoted for a 23.8 ms median. This walks
every warm pass of every rep of every arm and prints median [min-max] per
bucket, which is the form the numbers are evidence in.

WHAT IT DISCARDS: the first two `=== predictions ===` blocks of each rep, to
match `bench-decode.sh --cold 2` and the binary's own COLD_DECODE_STEPS. A cold
pass pays first-touch uploads and a cold page cache and belongs in no median.

THE ONE PARSING HAZARD, and why the split below exists: under
`INK_STAGE_SYNC=1` the report prints a `DEVICE per stage` block whose labels
COLLIDE with the host ones -- `attention half`, `shared experts` and `routed
experts` all appear twice at different indents. Reading the file top-to-bottom
therefore attributes device time to the host bucket on the stage-sync arm and
nowhere on the others, which reads as "the host got dearer" rather than as a
label collision. So the block is cut out by name, the `d_` keys are read from
inside it and everything else from outside it, and the head/unembed line (which
prints AFTER the block) is stitched back onto the outside half.

    scripts/stage-report-agg.py /tmp/moeA base,stagesync,stale,devplan

Numbers it prints are milliseconds per decode pass, and they inherit the
framing rule of the run that produced the logs -- layer range, context length,
lane, box, commit. Read them beside `bench-decode.sh`'s own framing block; a
bucket without it is not evidence.
"""
import re
import sys
import glob
import statistics

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/moeA"
arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base"]

# Keys starting `d_` are read from the "DEVICE per stage" block; every other key
# from the report with that block cut out. See the module docstring.
pats = {
    "pass_ms":     r"pass_ms ([0-9.]+)",
    "h_attn":      r"^      attention half\s+([0-9.]+)\s*$",
    "h_mlp":       r"^      mlp half\s+([0-9.]+)",
    "h_routed":    r"^        routed experts\s+([0-9.]+)",
    "h_shared":    r"^        shared experts\s+([0-9.]+)",
    "h_rest":      r"^        rest of half\s+([0-9.]+)",
    "h_router":    r"^      router \+ group\s+([0-9.]+)",
    "rt_mm":       r"matmul enqueue\s+([0-9.]+), BLOCKING read",
    "rt_read":     r"BLOCKING read\s+([0-9.]+), top-k",
    "rt_host":     r"top-k \+ group\s+([0-9.]+)",
    "stack_sync":  r"one sync for this node's whole stack:\s+([0-9.]+)",
    "d_attn":      r"^      attention half\s+([0-9.]+)\s*$",
    "d_router_mm": r"^      router matmul\s+([0-9.]+)",
    "d_routed":    r"^      routed experts\s+([0-9.]+)",
    "d_shared":    r"^      shared experts\s+([0-9.]+)",
    "d_tail":      r"^      sconv \+ resid\s+([0-9.]+)",
    "d_total":     r"^      staged total\s+([0-9.]+)",
    "head":        r"^    (?:head / unembed|tail \+ wire)\s+([0-9.]+)",
    "layer_loop":  r"^    layer loop\s+([0-9.]+)",
    "handback":    r"^      pool hand-back\s*([0-9.]+)",
    "after_sync":  r"^    after the sync\s+([0-9.]+)",
    "unattr":      r"^    UNATTRIBUTED\s+(-?[0-9.]+)",
    "stored_gib":  r"^    stored bytes\s+([0-9.]+) GiB",
}

DEVHDR = "DEVICE per stage"
HEADHDR = re.compile(r"^    (?:head / unembed|tail \+ wire)", re.M)


def collect(arm):
    vals = {k: [] for k in pats}
    for f in sorted(glob.glob("%s/%s.rep*.log" % (d, arm))):
        blocks = open(f, errors="replace").read().split("=== predictions ===")[1:]
        for b in blocks[2:]:
            if DEVHDR in b:
                pre, rest = b.split(DEVHDR, 1)
                m = HEADHDR.search(rest)
                dev = rest[: m.start()] if m else rest
                host = pre + (rest[m.start():] if m else "")
            else:
                dev, host = "", b
            for k, p in pats.items():
                hit = re.search(p, dev if k.startswith("d_") else host, re.M)
                if hit:
                    vals[k].append(float(hit.group(1)))
    return vals


def fmt(v):
    return "%8.2f [%4.1f-%4.1f]" % (statistics.median(v), min(v), max(v)) if v else "%18s" % "-"


data = {a: collect(a) for a in arms}
print("%-14s" % "metric" + "".join("%18s" % a for a in arms)
      + "   ms/pass, median [min-max] over warm passes")
for k in pats:
    print("%-14s" % k + "".join(fmt(data[a][k]) for a in arms))
print()
for a in arms:
    print(a, "warm passes sampled:", len(data[a]["pass_ms"]))
