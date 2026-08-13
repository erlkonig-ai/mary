#!/usr/bin/env python3
"""How much of a disagreement is the runtime disagreeing with ITSELF?

The NVFP4 runtime is not bit-reproducible. A single run per prompt is therefore
a SAMPLE of that model's answer, not the model's answer, and a paired number
built on single runs carries a noise term that has to be measured rather than
assumed small.

Measuring it is much cheaper than it looks, because a forward reports the top-5
at EVERY position, not just the one the harness scores. Running one prompt twice
yields as many paired comparisons as the prompt has tokens. Two runs of a
41-token prompt gave 41, and that is where the numbers below come from.

Three things are reported:

  * ARGMAX FLIP RATE over all positions — how often the same prompt through the
    same binary picks a different token. This is the floor under any paired
    disagreement rate: a difference between two models smaller than the
    runtime's difference with itself is not a difference between two models.
  * |delta top-1 logit| — how far the logits themselves move when nothing
    changes. Compare it with the BF16 reference's own batching wobble
    (`selfcheck_max_abs_logit_delta`); the larger of the two is what the
    comparison is really resolving to.
  * MARGIN AT FLIPS — flips should happen where the top two are close and
    nowhere else. If a flip ever lands on a wide margin, the wobble is not
    rounding and something is actually wrong.

The scored position specifically (the last one) is reported separately, because
that is the one the paired comparison uses and it is a single Bernoulli draw
per item, not per token.

  fp4_stability.py <items.json> <dir1> <dir2> [dir3 ...]
  fp4_stability.py --raw <rundir1> <rundir2>          (no item set needed)
"""
import json
import os
import re
import statistics
import sys

# `paired_score` is the sibling module, and this is spelled out because these
# scripts are run with `python -P` (the interpreter that does NOT put the
# script's own directory on the path) after a stray `/tmp/h2.py` in a shared
# scratch directory shadowed `h2` and broke every transformers import on the
# box. Safe path, explicit import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_score import read_fp4, wilson

LINE = re.compile(r"after token (\d+) \(id \d+\): top5 \[([^\]]*)\]\s+logits \[([^\]]*)\]")


def positions(path):
    """Every position's top-5 ids and logits, out of one tail log."""
    out = {}
    if not os.path.exists(path):
        return out
    for m in LINE.finditer(open(path, errors="replace").read()):
        out[int(m.group(1))] = ([int(x) for x in m.group(2).split(",")],
                                [float(x) for x in m.group(3).split(",")])
    return out


def compare(runs, label):
    """`runs` is a list of per-position dicts for the SAME prompt."""
    common = set(runs[0])
    for r in runs[1:]:
        common &= set(r)
    flips, deltas, flip_margins, keep_margins = [], [], [], []
    for k in sorted(common):
        tops = [r[k][0][0] for r in runs]
        m0 = runs[0][k][1][0] - runs[0][k][1][1]
        if len(set(tops)) > 1:
            flips.append(k)
            flip_margins.append(m0)
        else:
            keep_margins.append(m0)
        for r in runs[1:]:
            deltas.append(abs(runs[0][k][1][0] - r[k][1][0]))
    return {"label": label, "n": len(common), "flips": flips, "deltas": deltas,
            "flip_margins": flip_margins, "keep_margins": keep_margins,
            "last": [r[max(common)][0][0] for r in runs] if common else []}


def report(results):
    n = sum(r["n"] for r in results)
    nf = sum(len(r["flips"]) for r in results)
    deltas = [d for r in results for d in r["deltas"]]
    fm = [m for r in results for m in r["flip_margins"]]
    km = [m for r in results for m in r["keep_margins"]]
    last_stable = sum(1 for r in results if r["last"] and len(set(r["last"])) == 1)

    print()
    print(f"positions compared     {n} over {len(results)} prompt(s)")
    lo, hi = wilson(nf, n)
    print(f"argmax flip rate       {nf}/{n} = {100 * nf / n:.2f}% "
          f"[95% CI {100 * lo:.2f}–{100 * hi:.2f}]")
    if deltas:
        print(f"|delta top-1 logit|    median {statistics.median(deltas):.2f}, "
              f"max {max(deltas):.2f}  (BF16's own batching wobble was 0.875)")
    if fm:
        print(f"margin where it flipped   max {max(fm):.2f}, median "
              f"{statistics.median(fm):.2f}")
    if km:
        print(f"margin where it did not   median {statistics.median(km):.2f}")
    if fm and km and max(fm) < statistics.median(km):
        print("  -> every flip is on a margin narrower than the typical position, which is "
              "what rounding looks like and not what a bug looks like")
    slo, shi = wilson(last_stable, len(results))
    print(f"SCORED position stable {last_stable}/{len(results)} = "
          f"{100 * last_stable / max(len(results), 1):.1f}% "
          f"[95% CI {100 * slo:.1f}–{100 * shi:.1f}]")
    print()
    print("Read the scored-position line as the floor under the paired result: a loss")
    print("rate below the rate at which the runtime disagrees with itself is not evidence")
    print("about quantisation.")


def main():
    if sys.argv[1] == "--raw":
        runs = [positions(os.path.join(d, "tail.log")) for d in sys.argv[2:]]
        assert len(runs) >= 2, "two or more run directories"
        r = compare(runs, "raw")
        print(f"{'position':>9}  " + "  ".join(f"{os.path.basename(d):>12}" for d in sys.argv[2:]))
        for k in sorted(set.intersection(*[set(x) for x in runs])):
            tops = [x[k][0][0] for x in runs]
            mark = "" if len(set(tops)) == 1 else "   <-- FLIPPED"
            if mark:
                print(f"{k:>9}  " + "  ".join(f"{t:>12}" for t in tops) + mark)
        report([r])
        return

    items = json.load(open(sys.argv[1]))["items"]
    dirs = sys.argv[2:]
    assert len(dirs) >= 2, "two or more run directories"
    results = []
    print(f"{'item':<16} {'family':<10} " + "  ".join(f"{os.path.basename(d):>10}" for d in dirs)
          + "   positions  flips")
    for it in items:
        runs, ok = [], True
        for d in dirs:
            rd = os.path.join(d, it["key"])
            if not os.path.isdir(rd):
                ok = False
                break
            read_fp4(rd, it["ids"])          # the prompt.ids guard, for its side effect
            runs.append(positions(os.path.join(rd, "tail.log")))
        if not ok or any(not r for r in runs):
            continue
        r = compare(runs, it["key"])
        results.append(r)
        marks = "  ".join(f"{x:>10}" for x in r["last"])
        print(f"{it['key']:<16} {it['family']:<10} {marks}   {r['n']:>9}  {len(r['flips']):>5}"
              f"{'   <-- SCORED TOKEN FLIPPED' if len(set(r['last'])) > 1 else ''}")
    if not results:
        raise SystemExit("no item was present in every directory")
    report(results)


if __name__ == "__main__":
    main()
