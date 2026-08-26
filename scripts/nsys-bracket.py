#!/usr/bin/env python3
"""nsys-bracket.py -- bracket a WARM DECODE STEP into device busy, device idle
and host driver time, from ONE nsys profile, by cutting the timeline at a kernel
that fires exactly once per step.

WHY NOT THE OBVIOUS METHOD. The tempting way to get a per-step figure out of a
profile whose totals are dominated by setup is to run the same config twice at
two INK_GEN values and difference them: the index build, the 82.7 GiB weight
copy and the kernel JIT are identical, so (hi - lo) / (gen_hi - gen_lo) should
be per-step with every fixed cost removed. IT IS NOT RELIABLE HERE, measured:
the PREFILL pass is 3732 rows wide and costs ~1.3 s of `w4a16_linear` on its
own, and it does not reproduce between two runs of the same binary to better
than ~10%. On the head-only config that irreproducibility was 51 ms (off arm)
and 130 ms (rule arm) on one kernel; spread over 20 steps that is 2.5 and 6.5
ms/step of ERROR on a quantity whose true value is ~6 and ~5 ms/step. The
difference reported the swizzled kernel as 2.9x SLOWER when a warm-window count
on the same two profiles shows it 1.30x FASTER. A difference of two numbers each
carrying an unreproducible 1.3 s term is not a small-number measurement.

SO CUT THE TIMELINE INSTEAD. `--anchor` names a kernel that launches exactly
once per decode step (`ann_scan_kernel`, the approximate head's scan, is the
default and is one per step on any node that owns the unembed). Consecutive
anchor starts bound one whole step, device-side. The last `--last N` such
intervals are warm by construction -- they are the end of the run, past every
cold pass -- and the median over them is reported with its spread, so no figure
here rests on one step.

WHAT EACH NUMBER IS PER. PER DECODE STEP, ONE PROCESS (one node), on the config
the profile's own metadata names (echoed below, so a figure cannot be separated
from its layer range). Not per token unless tokens/pass is 1, and not per layer.

  step period    anchor to anchor. The device-side step; it agrees with the
                 binary's own ms/step when the run is not stalled.
  device busy    UNION of GPU activity intervals inside the step (kernels +
                 memcpy), so concurrent streams count once. What the device
                 would take with a zero-cost host.
  device idle    step period - device busy. Time the GPU had nothing to run,
                 i.e. the room a host-side saving has to move into.
  driver calls   wall time the LAUNCHER thread spends inside CUDA driver calls,
                 split into enqueue-ish calls and BLOCKING waits. A blocking
                 wait on the launcher thread is device time inside whatever
                 host bracket happens to be open -- which is why the binary's
                 "HOST, enqueue only" lines move when only a memory LAYOUT
                 changed.

USAGE
  scripts/nsys-bracket.py PROFILE.sqlite [PROFILE.sqlite ...]
        [--anchor KERNEL] [--last N] [--top K]

  .sqlite comes from `nsys export --type sqlite <rep>.nsys-rep`.
  Several profiles may be given; each is bracketed and the table is comparable
  row to row ONLY when the profiles differ in one thing and say so.
"""

import sqlite3
import sys
import argparse
import statistics

WAIT_CALLS = {
    "cuEventSynchronize",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuMemcpyDtoH_v2",
    "cuMemcpyHtoD_v2",
}


def union_ms(rows, lo=None, hi=None):
    """Union length of [start,end) intervals clipped to [lo,hi), in ms."""
    iv = []
    for s, e in rows:
        if lo is not None:
            s = max(s, lo)
            e = min(e, hi)
        if e > s:
            iv.append((s, e))
    if not iv:
        return 0.0
    iv.sort()
    total = 0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    total += ce - cs
    return total / 1e6


def analyse(path, anchor, last, top):
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
    t0, t1 = anchors[0], anchors[-1]

    kern = db.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName=s.id WHERE k.start>=? AND k.start<?",
        (t0, t1),
    ).fetchall()
    try:
        cpy = db.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE start>=? AND start<?",
            (t0, t1),
        ).fetchall()
    except sqlite3.OperationalError:
        cpy = []

    lt = db.execute(
        "SELECT r.globalTid, count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId=s.id WHERE s.value='cuLaunchKernel' "
        "GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
    ).fetchone()
    launcher = lt[0] if lt else None
    api = db.execute(
        "SELECT r.start, r.end, s.value, r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId=s.id WHERE r.start>=? AND r.start<?",
        (t0, t1),
    ).fetchall()
    db.close()

    periods, busy, enq, wait, waito = [], [], [], [], []
    kacc = {}
    for i in range(last):
        a, b = anchors[i], anchors[i + 1]
        periods.append((b - a) / 1e6)
        iv = [(s, e) for s, e, _ in kern if s < b and e > a] + [
            (s, e) for s, e in cpy if s < b and e > a
        ]
        busy.append(union_ms(iv, a, b))
        q = w = wo = 0.0
        for s, e, n, tid in api:
            if s < a or s >= b:
                continue
            d = (e - s) / 1e6
            if n in WAIT_CALLS:
                if tid == launcher:
                    w += d
                else:
                    wo += d
            elif tid == launcher:
                q += d
        enq.append(q)
        wait.append(w)
        waito.append(wo)
        for s, e, n in kern:
            if a <= s < b:
                kacc.setdefault(n, [0, 0.0])
                kacc[n][0] += 1
                kacc[n][1] += (e - s) / 1e6

    def med(v):
        return statistics.median(v)

    def spread(v):
        return 100.0 * (max(v) - min(v)) / med(v) if med(v) else 0.0

    print(f"=== {path} ===")
    print(f"  config : {' '.join(meta)}")
    print(
        f"  window : the last {last} intervals between consecutive '{anchor}' "
        f"launches (one decode step each)"
    )
    print(f"  {'step period':<14} {med(periods):8.3f} ms/step   (spread {spread(periods):.1f}% over {last} steps)")
    print(f"  {'device busy':<14} {med(busy):8.3f} ms/step   ({100*med(busy)/med(periods):.0f}% of the step; union of kernel+memcpy)")
    print(f"  {'device idle':<14} {med(periods)-med(busy):8.3f} ms/step   (the GPU had nothing to run)")
    print(f"  {'driver enqueue':<14} {med(enq):8.3f} ms/step   (launcher thread, non-blocking calls)")
    print(f"  {'driver BLOCK':<14} {med(wait):8.3f} ms/step   (launcher thread, blocking waits = device time")
    print(f"  {'':<14} {'':8}              inside whatever host bracket is open)")
    print(f"  {'other threads':<14} {med(waito):8.3f} ms/step   (blocking waits the enqueue never sees)")
    print(f"  --- top {top} kernels, per decode step ---")
    rows = sorted(((v[1] / last, v[0] / last, n) for n, v in kacc.items()), reverse=True)
    for ms, cnt, n in rows[:top]:
        print(f"  {ms:8.3f} ms/step  {cnt:7.1f} launches/step  {n[:66]}")
    print()
    return {
        "period": med(periods),
        "busy": med(busy),
        "enq": med(enq),
        "wait": med(wait),
        "kern": {n: (v[0] / last, v[1] / last) for n, v in kacc.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="+")
    ap.add_argument("--anchor", default="ann_scan_kernel")
    ap.add_argument("--last", type=int, default=8)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    for p in a.profiles:
        analyse(p, a.anchor, a.last, a.top)


if __name__ == "__main__":
    main()
