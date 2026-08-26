#!/usr/bin/env python3
"""nsys-bracket.py -- bracket a decode step into HOST ENQUEUE, DEVICE BUSY and
EXPOSED DEVICE TIME, by DIFFERENCING two nsys profiles of the same config that
differ only in INK_GEN.

WHY A DIFFERENCE AND NOT A TOTAL. A single profile of one run is dominated by
setup: index build, the 82.7 GiB weight copy, kernel JIT, arena warm-up. On the
head-only config `cuMemcpy2DAsync_v2` alone is 1.30 s of a profile whose entire
decode is 0.6 s. Nothing per-step can be read off it. Two runs of the SAME
binary, SAME prompt, SAME layers and DIFFERENT INK_GEN share that setup exactly,
so (g_hi - g_lo) / (hi - lo) is a per-decode-step figure with every fixed cost
subtracted. This is the method the graph lane used to derive its per-node device
figures; this script is that method written down so the next reader runs it
rather than re-deriving it.

WHAT EACH NUMBER IS PER. Every figure this prints is PER DECODE STEP, for ONE
PROCESS (one node), on the config named in the profile's own metadata (which it
echoes, so a figure can never be separated from the layer range and INK_KV that
produced it). It is NOT per token unless tokens/pass is 1, and NOT per layer.

  device busy    the UNION of GPU activity intervals (kernels + memcpy), so
                 concurrent streams are counted once. This is what the device
                 would take with a zero-cost host.
  kernel sum     the SUM of kernel durations. Above `device busy` only when
                 streams overlap; on this decode lane they are within 1%.
  host enqueue   wall time the launching thread spends inside CUDA driver calls
                 (cuLaunchKernel, cuMemcpy*Async, cuMemAlloc*, ...). It excludes
                 blocking waits (cuEventSynchronize, cuCtxSynchronize,
                 cuStreamSynchronize) -- those are the host WAITING FOR the
                 device, not enqueueing, and folding them in would double-count
                 device time as host cost.
  host wait      the blocking waits, reported separately for exactly that reason.
  step           the run's own reported ms/step, if given with --step-lo/--step-hi.

  exposed device = step - host enqueue - host wait.  The device time that is NOT
                 hidden behind the host. It is the ONLY part of a device-side
                 win that a step can see, which is the whole point of the
                 bracket.

USAGE
  scripts/nsys-bracket.py LO.sqlite HI.sqlite [--top N] [--step-lo MS --step-hi MS]

The two .sqlite files come from `nsys export --type sqlite <rep>.nsys-rep`.
"""

import sqlite3
import sys
import argparse

# Driver calls that are the host WAITING on the device, not enqueueing work.
WAIT_CALLS = {
    "cuEventSynchronize",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuMemcpyDtoH_v2",
    "cuMemcpyHtoD_v2",
}


def meta(db):
    cur = db.execute(
        "SELECT name, value FROM META_DATA_CAPTURE "
        "WHERE name LIKE 'PROCESS_0:ARGUMENT%' OR name='PROCESS_0:COMMAND' "
        "ORDER BY name"
    )
    return [v for _, v in cur.fetchall()]


def union_ms(rows):
    """Total length of the union of [start, end) intervals, in ms."""
    if not rows:
        return 0.0
    rows = sorted(rows)
    total = 0
    cs, ce = rows[0]
    for s, e in rows[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    total += ce - cs
    return total / 1e6


def profile(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {"meta": meta(db)}

    iv = db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    try:
        iv += db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    except sqlite3.OperationalError:
        pass
    try:
        iv += db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET").fetchall()
    except sqlite3.OperationalError:
        pass
    out["device_busy"] = union_ms(iv)

    krows = db.execute(
        "SELECT s.value, count(*), sum(k.end-k.start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "GROUP BY 1"
    ).fetchall()
    out["kernels"] = {n: (c, ms) for n, c, ms in krows}
    out["kernel_sum"] = sum(ms for _, ms in out["kernels"].values())
    out["kernel_n"] = sum(c for c, _ in out["kernels"].values())

    rrows = db.execute(
        "SELECT s.value, count(*), sum(r.end-r.start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
        "GROUP BY 1"
    ).fetchall()
    out["api"] = {n: (c, ms) for n, c, ms in rrows}

    # WHICH THREAD BLOCKS MATTERS, and it is the difference between two
    # completely different readings of the same total. A blocking wait on the
    # thread that also issues `cuLaunchKernel` is the model's own enqueue loop
    # stopping dead -- device time showing up inside a bracket labelled "host,
    # enqueue only". The same wait on any other thread is a helper the enqueue
    # never sees. So the launcher thread is identified by what it does, not by
    # its id (ids differ between runs and cannot be paired), and the wait time is
    # split on that.
    lt = db.execute(
        "SELECT r.globalTid, count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId=s.id WHERE s.value='cuLaunchKernel' "
        "GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
    ).fetchone()
    launcher = lt[0] if lt else None
    out["launcher_tid"] = launcher
    wl = wo = 0.0
    for name, tid, ms in db.execute(
        "SELECT s.value, r.globalTid, sum(r.end-r.start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
        "GROUP BY 1,2"
    ).fetchall():
        if name not in WAIT_CALLS:
            continue
        if tid == launcher:
            wl += ms
        else:
            wo += ms
    out["wait_launcher"] = wl
    out["wait_other"] = wo
    out["host_enqueue"] = sum(
        ms for n, (_, ms) in out["api"].items() if n not in WAIT_CALLS
    )
    out["host_wait"] = sum(
        ms for n, (_, ms) in out["api"].items() if n in WAIT_CALLS
    )
    out["api_n"] = sum(c for c, _ in out["api"].values())
    db.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo")
    ap.add_argument("hi")
    ap.add_argument("--gen-lo", type=int, default=None)
    ap.add_argument("--gen-hi", type=int, default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--step-lo", type=float, default=None,
                    help="ms/step the LO run reported (optional, for `exposed`)")
    ap.add_argument("--step-hi", type=float, default=None)
    a = ap.parse_args()

    lo, hi = profile(a.lo), profile(a.hi)

    def gen_of(p, fallback):
        for v in p["meta"]:
            if v.startswith("INK_GEN="):
                return int(v.split("=", 1)[1])
        return fallback

    glo = a.gen_lo if a.gen_lo else gen_of(lo, 0)
    ghi = a.gen_hi if a.gen_hi else gen_of(hi, 0)
    n = ghi - glo
    if n <= 0:
        sys.exit(f"!! INK_GEN must differ and hi>lo (got {glo} and {ghi})")

    print(f"=== per-decode-step bracket, ONE PROCESS, by (GEN {ghi} - GEN {glo}) / {n} ===")
    print(f"  lo : {a.lo}")
    print(f"       {' '.join(lo['meta'][1:])}")
    print(f"  hi : {a.hi}")
    print(f"       {' '.join(hi['meta'][1:])}")
    if lo["meta"][1:] != hi["meta"][1:]:
        diff = [x for x in lo["meta"] if x not in hi["meta"]] + [
            x for x in hi["meta"] if x not in lo["meta"]
        ]
        nong = [x for x in diff if not x.startswith("INK_GEN=") and not x.endswith(".bin")]
        if nong:
            print(f"  !! THE TWO RUNS DIFFER IN MORE THAN INK_GEN: {nong}")
            print("     A difference is only a per-step figure when everything else is shared.")
    print()

    def d(k):
        return (hi[k] - lo[k]) / n

    print(f"  {'device busy':<16} {d('device_busy'):8.3f} ms/step   (union of kernel+memcpy intervals)")
    print(f"  {'kernel sum':<16} {d('kernel_sum'):8.3f} ms/step   ({(hi['kernel_n']-lo['kernel_n'])/n:.0f} kernels/step)")
    print(f"  {'host enqueue':<16} {d('host_enqueue'):8.3f} ms/step   ({(hi['api_n']-lo['api_n'])/n:.0f} driver calls/step)")
    print(f"  {'host wait':<16} {d('host_wait'):8.3f} ms/step   (blocking sync, host waiting ON the device)")
    print(f"  {'  on launcher':<16} {d('wait_launcher'):8.3f} ms/step   (the SAME thread that issues cuLaunchKernel:")
    print(f"  {'':<16} {'':8}              this is device time inside the enqueue bracket)")
    print(f"  {'  on others':<16} {d('wait_other'):8.3f} ms/step   (helper threads; the enqueue never sees these)")
    if a.step_lo and a.step_hi:
        step = (a.step_hi * ghi - a.step_lo * glo) / n
        print(f"  {'step':<16} {step:8.3f} ms/step   (from the runs' own WARM medians, same difference)")
        exposed = step - d("host_enqueue") - d("host_wait")
        print(f"  {'EXPOSED device':<16} {exposed:8.3f} ms/step   = step - host enqueue - host wait")
        print(f"  {'hidden device':<16} {d('device_busy')-exposed:8.3f} ms/step   = device busy - exposed")
        print(f"  {'host-bound frac':<16} {100*(d('host_enqueue')+d('host_wait'))/step:7.1f}%    of the step is host")
    print()

    print(f"  --- top {a.top} kernels, per decode step ---")
    rows = []
    for name in set(hi["kernels"]) | set(lo["kernels"]):
        c1, m1 = hi["kernels"].get(name, (0, 0.0))
        c0, m0 = lo["kernels"].get(name, (0, 0.0))
        rows.append((( m1 - m0) / n, (c1 - c0) / n, name))
    rows.sort(reverse=True)
    for ms, cnt, name in rows[: a.top]:
        print(f"  {ms:8.3f} ms/step  {cnt:7.1f} launches/step  {name[:70]}")
    print()

    print(f"  --- top driver calls, per decode step ---")
    rows = []
    for name in set(hi["api"]) | set(lo["api"]):
        c1, m1 = hi["api"].get(name, (0, 0.0))
        c0, m0 = lo["api"].get(name, (0, 0.0))
        rows.append(((m1 - m0) / n, (c1 - c0) / n, name))
    rows.sort(reverse=True)
    for ms, cnt, name in rows[:10]:
        tag = "  [WAIT]" if name in WAIT_CALLS else ""
        print(f"  {ms:8.3f} ms/step  {cnt:7.1f} calls/step  {name}{tag}")


if __name__ == "__main__":
    main()
