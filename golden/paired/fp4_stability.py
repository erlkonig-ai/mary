#!/usr/bin/env python3
"""How much of a disagreement is the runtime disagreeing with ITSELF?

The NVFP4 runtime is not bit-reproducible: about one pass in fifteen diverges,
at around 1.5% of positions. A single run per prompt is therefore a SAMPLE of
that model's answer, not the model's answer, and any paired number built on
single runs carries an unquantified term.

This measures the term. Give it the same items run twice (or more) into
different directories and it reports, per item, whether the greedy answer token
was stable, and in aggregate the fraction of items whose answer is NOT a
property of the model but of the run.

That number is what belongs in the uncertainty on the paired result: if r items
in n flip between two runs of the SAME model, then a disagreement rate measured
against a different model carries at least that much irreducible noise, and a
loss smaller than it is not evidence of anything.

Re-running everything doubles hours of compute, so the honest cheap version is
to re-run a SUBSET chosen to include every item the two models disagreed on —
those are the items sitting closest to a decision boundary and therefore the
ones most able to flip. A stability rate measured on the disagreements is a
conservative (pessimistic) estimate for the set as a whole, which is the right
direction for an error bar to lean.

  fp4_stability.py <items.json> <dir1> <dir2> [dir3 ...]
"""
import json
import math
import os
import sys

from paired_score import read_fp4, wilson


def main():
    items = json.load(open(sys.argv[1]))["items"]
    dirs = sys.argv[2:]
    assert len(dirs) >= 2, "two or more run directories"

    by_key = {it["key"]: it for it in items}
    n_cmp = n_stable = 0
    print(f"{'item':<16} {'family':<10} " + " ".join(f"{os.path.basename(d):>10}" for d in dirs))
    for key, it in by_key.items():
        got = []
        for d in dirs:
            rd = os.path.join(d, key)
            got.append(read_fp4(rd, it["ids"]) if os.path.isdir(rd) else None)
        if any(g is None for g in got):
            continue
        n_cmp += 1
        same = len(set(g["argmax"] for g in got)) == 1
        n_stable += same
        marks = " ".join(f"{g['argmax']:>10}" for g in got)
        print(f"{key:<16} {it['family']:<10} {marks}   {'' if same else '<-- FLIPPED'}")

    if n_cmp == 0:
        raise SystemExit("no item was present in every directory")
    lo, hi = wilson(n_stable, n_cmp)
    print()
    print(f"stable across {len(dirs)} runs: {n_stable}/{n_cmp} = {100 * n_stable / n_cmp:.1f}% "
          f"[95% CI {100 * lo:.1f}–{100 * hi:.1f}]")
    flo, fhi = wilson(n_cmp - n_stable, n_cmp)
    print(f"per-item flip rate         : {n_cmp - n_stable}/{n_cmp} = "
          f"{100 * (n_cmp - n_stable) / n_cmp:.1f}% [95% CI {100 * flo:.1f}–{100 * fhi:.1f}]")
    print()
    print("Read this as the floor under any paired disagreement: a difference between")
    print("the two models smaller than the runtime's difference with itself is not a")
    print("difference between the two models.")


if __name__ == "__main__":
    main()
