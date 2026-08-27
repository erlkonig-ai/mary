#!/usr/bin/env python3
"""nsys-memcpy-census.py -- how many memcpy operations a warm decode step
issues, and how big they are, cut out of ONE nsys profile by the same anchor
`nsys-bracket.py` uses.

WHY A COUNT AND NOT A TIME. The thing under test here -- cubecl's per-launch
dynamic-metadata upload, and the cache that removes it -- changes how many
host-to-device copies a step makes. A COUNT is exact: it does not need reps, it
carries no variance, and it cannot be confused by a box that was busy. The
paired two-node step, by contrast, cannot resolve a device change smaller than
about 1 ms/node at 7 reps, so any timing claim from it has to be stated with
that resolution beside it. This instrument deliberately reports the quantity
that does not need one.

WHAT EACH NUMBER IS PER. PER DECODE STEP, ONE PROCESS (one node), on the config
the profile's own metadata names, which is echoed so a figure cannot be
separated from it. Consecutive launches of the anchor kernel bound one step
device-side; the last `--last N` such intervals are warm by construction, and
every count is the MEDIAN over them with the min and max beside it, so no figure
rests on one step.

USAGE
  scripts/nsys-memcpy-census.py PROFILE.sqlite [PROFILE.sqlite ...]
        [--anchor KERNEL] [--last N]

  .sqlite comes from `nsys export --type sqlite <rep>.nsys-rep`.
  Several profiles may be given; the rows are comparable ONLY when the profiles
  differ in one thing and say so.
"""

import argparse
import sqlite3
import statistics
import sys

# CUPTI's copyKind enum. Only the ones a decode step can produce are named; an
# unknown value is printed as its number rather than guessed at.
COPY_KIND = {
    1: "HtoD",
    2: "DtoH",
    3: "HtoA",
    4: "AtoH",
    5: "AtoA",
    6: "AtoD",
    7: "DtoA",
    8: "DtoD",
    9: "HtoH",
    10: "PtoP",
}


def census(path, anchor, last):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    meta = [
        v
        for _, v in db.execute(
            "SELECT name, value FROM META_DATA_CAPTURE "
            "WHERE name LIKE 'PROCESS_0:ARGUMENT%' ORDER BY name"
        ).fetchall()
    ]

    anchors = [
        r[0]
        for r in db.execute(
            "SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName=s.id WHERE s.value=? ORDER BY k.start",
            (anchor,),
        ).fetchall()
    ]
    if len(anchors) < last + 1:
        sys.exit(
            f"!! {path}: only {len(anchors)} launches of anchor '{anchor}'; "
            f"--last {last} needs {last + 1}. Name a kernel that fires once per step."
        )
    anchors = anchors[-(last + 1):]

    try:
        cpy = db.execute(
            "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE start>=? AND start<? ORDER BY start",
            (anchors[0], anchors[-1]),
        ).fetchall()
    except sqlite3.OperationalError:
        cpy = []
    kern = [
        r[0]
        for r in db.execute(
            "SELECT start FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start>=? AND start<? ORDER BY start",
            (anchors[0], anchors[-1]),
        ).fetchall()
    ]

    # One bucket per step, so what is reported is a median over steps and not a
    # total divided by a step count -- a total hides a step that differed.
    steps = []
    for i in range(len(anchors) - 1):
        lo, hi = anchors[i], anchors[i + 1]
        rows = [(b, k, e - s) for (s, e, b, k) in cpy if lo <= s < hi]
        steps.append(
            {
                "period_ns": hi - lo,
                "copies": len(rows),
                "htod": sum(1 for (_, k, _d) in rows if k == 1),
                "htod_bytes": sum(b for (b, k, _d) in rows if k == 1),
                # A SUM of durations, not an occupancy: two copies on different
                # engines can overlap, so this is an upper bound on what they
                # hold the device for. `nsys-bracket.py` gives the union.
                "htod_ns": sum(d for (_b, k, d) in rows if k == 1),
                "kernels": sum(1 for s in kern if lo <= s < hi),
                "sizes": [b for (b, k, _d) in rows if k == 1],
                "kinds": [k for (_b, k, _d) in rows],
            }
        )

    def med(key):
        v = [s[key] for s in steps]
        return statistics.median(v), min(v), max(v)

    print(f"\n=== {path}")
    if meta:
        print("  argv: " + " ".join(meta))
    print(
        f"  anchor '{anchor}', {len(steps)} warm step intervals "
        f"(median, min, max over them)"
    )
    for label, key in (
        ("step period ns  ", "period_ns"),
        ("kernel launches ", "kernels"),
        ("memcpy ops      ", "copies"),
        ("  of them HtoD  ", "htod"),
        ("HtoD bytes      ", "htod_bytes"),
        ("HtoD device ns  ", "htod_ns"),
    ):
        m, lo, hi = med(key)
        print(f"  {label}  {m:>12,.0f}   [{lo:,} .. {hi:,}]")
    mp, _, _ = med("period_ns")
    mh, _, _ = med("htod_ns")
    if mp:
        print(
            f"  the HtoD copies sum to {100.0 * mh / mp:.2f}% of the step period "
            f"(a SUM, so an upper bound on their occupancy)"
        )

    kinds = {}
    for s in steps:
        for k in s["kinds"]:
            kinds[k] = kinds.get(k, 0) + 1
    if kinds:
        n = len(steps)
        print("  by copy kind, per step:")
        for k in sorted(kinds):
            print(f"    {COPY_KIND.get(k, k):<6} {kinds[k] / n:>8.1f}")

    # The size histogram is what makes this comparable to the census the design
    # note carries: 16 B x306, 32 B x103, 64 B x57, 144 B x2, 208 B x36.
    hist = {}
    for s in steps:
        for b in s["sizes"]:
            hist[b] = hist.get(b, 0) + 1
    if hist:
        n = len(steps)
        print("  HtoD size histogram, per step:")
        for b in sorted(hist):
            print(f"    {b:>8,} B   x{hist[b] / n:>8.1f}")
    else:
        print("  HtoD size histogram, per step:  (none)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="+")
    ap.add_argument("--anchor", default="ann_scan_kernel")
    ap.add_argument("--last", type=int, default=12)
    a = ap.parse_args()
    for p in a.profiles:
        census(p, a.anchor, a.last)


if __name__ == "__main__":
    main()
