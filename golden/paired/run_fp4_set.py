#!/usr/bin/env python3
"""Drive the NVFP4 runtime over a whole item set, one head+tail pair per item.

`inkling_forward` answers one ids file per process and there is no batch mode,
so an item set is N sequential pairs and the per-item cost is dominated by
process start: each half warms its share of the expert slabs through the page
cache before it computes anything. Measured on spark, 41 tokens: 4m14s wall,
of which about 2m20s is the tail warming before it will even accept the head's
connection.

That is why this script is RESUMABLE. A set of sixty items is hours; a session
that dies at item forty should not start again at item one. An item is skipped
when its directory already holds a tail log with a final-position line AND a
`prompt.ids` equal to the item's own — the second condition matters, because
the cheap way to get a wrong answer here is to reuse a directory that answers a
different prompt.

  run_fp4_set.py <items.json> <outdir> [--only k1,k2] [--limit N] [--force]
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r"after token (\d+) \(id \d+\): top5")


def done(rundir, ids):
    idsf = os.path.join(rundir, "prompt.ids")
    log = os.path.join(rundir, "tail.log")
    if not (os.path.exists(idsf) and os.path.exists(log)):
        return False
    raw = open(idsf, "rb").read()
    got = list(struct.unpack("<%dq" % (len(raw) // 8), raw))
    if got != list(ids):
        return False
    hits = [int(m.group(1)) for m in LINE.finditer(open(log, errors="replace").read())]
    return bool(hits) and max(hits) == len(ids) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("outdir")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--port", default="7654")
    args = ap.parse_args()

    items = json.load(open(args.items))["items"]
    if args.only:
        want = set(args.only.split(","))
        items = [it for it in items if it["key"] in want]
    if args.limit:
        items = items[: args.limit]
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    for n, it in enumerate(items, 1):
        rundir = os.path.join(args.outdir, it["key"])
        if not args.force and done(rundir, it["ids"]):
            print(f"[{n}/{len(items)}] {it['key']:<16} already done", flush=True)
            continue
        os.makedirs(rundir, exist_ok=True)
        idsf = os.path.join(rundir, "in.ids")
        with open(idsf, "wb") as fh:
            fh.write(struct.pack("<%dq" % len(it["ids"]), *it["ids"]))
        t = time.time()
        rc = subprocess.call([os.path.join(HERE, "run_fp4.sh"), idsf, rundir, args.port])
        el = time.time() - t
        ok = done(rundir, it["ids"])
        print(f"[{n}/{len(items)}] {it['key']:<16} rc={rc} {el:6.1f}s "
              f"{'ok' if ok else 'INCOMPLETE'}   elapsed {(time.time() - t0) / 60:.1f} min",
              flush=True)
        if rc != 0 and not ok:
            print(f"  see {rundir}/tail.log and {rundir}/head.log", flush=True)
    print(f"set done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
