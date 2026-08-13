#!/usr/bin/env python3
"""The paired comparison itself: where BF16 and NVFP4 DISAGREE, not what either scores.

# Why paired

An absolute benchmark number on forty items has a 95% interval about fifteen
points wide, which is wider than any quantisation effect worth finding. The
same forty items through both models is a different measurement: the item's
difficulty, its contamination, its ambiguity and its answer-key are all shared
by the two arms and cancel. What is left is the thing we own both sides of.

So the headline is not an accuracy. It is:

  * AGREEMENT  — the two models' greedy answer token, identical or not. This is
    the most sensitive number here and it needs no answer key at all.
  * LOSS       — of the items BF16 gets right, the fraction NVFP4 gets wrong.
    This is the one that says whether quantisation cost capability.
  * CLUSTERING — whether the losses fall in one family more than chance allows.
    A loss concentrated in `arith` is a statement about what broke; the same
    number spread evenly across five families is noise with a story attached.

McNemar's exact test is the right test for LOSS because it conditions on the
discordant pairs, which is exactly what "we own both models" gives you.

# Counts, always

Every rate is printed as k/n with a Wilson 95% interval. A percentage without
its denominator is how "75% from forty events" turned out to be 45.9% over two
thousand, in this project, three weeks ago.

# It can fail, and here is the switch that makes it

`--flip K` corrupts K of the NVFP4 answers before scoring — a synthetic
regression of known size — and prints what the harness then says. Run it to see
the loss rate leave its interval and the McNemar p collapse, and to find out how
big a regression this many items can actually detect. A check nobody has
watched fail is not evidence.

  paired_score.py <items.json> <ref.json> <fp4dir> [options]
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys


# --------------------------------------------------------------------------
# statistics, hand-rolled: this venv has numpy and torch, not scipy, and all
# three of these are short enough that a dependency would be the bigger risk.
# --------------------------------------------------------------------------
def wilson(k, n, z=1.959963985):
    """95% Wilson score interval. Correct at k=0 and k=n, which is where a
    normal-approximation interval leaves the unit interval entirely."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_rate(k, n, label=""):
    lo, hi = wilson(k, n)
    if n == 0:
        return f"{label}n/a (0 items)"
    return f"{label}{k}/{n} = {100 * k / n:.1f}%  [95% CI {100 * lo:.1f}–{100 * hi:.1f}]"


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: P(|X - (b+c)/2| >= |b - (b+c)/2|), X ~ Bin(b+c, 1/2).

    Conditions on the discordant pairs only. The concordant items -- both right
    or both wrong -- carry no information about a DIFFERENCE between the arms,
    which is the whole reason the paired form is more sensitive than two
    accuracies.
    """
    n = b + c
    if n == 0:
        return 1.0
    obs = abs(b - n / 2)
    tot = 0
    for k in range(n + 1):
        if abs(k - n / 2) >= obs - 1e-12:
            tot += math.comb(n, k)
    return min(1.0, tot / 2 ** n)


def cluster_test(labels, lost, draws=20000, seed=20260813):
    """p-value and observed chi-square-like statistic for uneven losses."""
    n = len(labels)
    hits = sum(lost)
    fams = sorted(set(labels))
    idx = {f: i for i, f in enumerate(fams)}
    size = [0] * len(fams)
    for g in labels:
        size[idx[g]] += 1

    def stat(counts):
        s = 0.0
        for i in range(len(fams)):
            e = hits * size[i] / n
            if e > 0:
                s += (counts[i] - e) ** 2 / e
        return s

    obs_counts = [0] * len(fams)
    for g, l in zip(labels, lost):
        if l:
            obs_counts[idx[g]] += 1
    obs = stat(obs_counts)
    if hits == 0 or hits == n:
        return 1.0, obs

    rng = random.Random(seed)
    pool = list(range(n))
    ge = 0
    for _ in range(draws):
        pick = rng.sample(pool, hits)
        counts = [0] * len(fams)
        for j in pick:
            counts[idx[labels[j]]] += 1
        if stat(counts) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (draws + 1), obs


# --------------------------------------------------------------------------
# reading the two sides
# --------------------------------------------------------------------------
LINE = re.compile(r"after token (\d+) \(id (\d+)\): top5 \[([^\]]*)\]\s+logits \[([^\]]*)\]")


def read_fp4(rundir, expect_ids):
    """One NVFP4 run: the last position's top-5 ids and logits, out of the tail log.

    The ids the run actually consumed are re-read from the run directory and
    compared with the item's, because an ids file that has drifted from the
    text somebody thinks it holds is a failure this tree has already had once.
    A mismatch is fatal here, not a warning.
    """
    idsf = os.path.join(rundir, "prompt.ids")
    if os.path.exists(idsf):
        raw = open(idsf, "rb").read()
        got = [int.from_bytes(raw[i:i + 8], "little", signed=True) for i in range(0, len(raw), 8)]
        if got != list(expect_ids):
            raise SystemExit(f"{rundir}: prompt.ids is not this item's prompt "
                             f"({len(got)} ids vs {len(expect_ids)})")
    log = os.path.join(rundir, "tail.log")
    if not os.path.exists(log):
        return None
    best = None
    for m in LINE.finditer(open(log, errors="replace").read()):
        pos = int(m.group(1))
        if best is None or pos > best[0]:
            best = (pos,
                    [int(x) for x in m.group(3).split(",")],
                    [float(x) for x in m.group(4).split(",")])
    if best is None:
        return None
    return {"pos": best[0], "top_ids": best[1], "top_logits": best[2], "argmax": best[1][0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("ref")
    ap.add_argument("fp4dir")
    # `open` by default: the sealed set is cheap to RUN and expensive to look
    # at, so looking at it is the thing that takes a flag.
    ap.add_argument("--split", default="open", choices=["all", "open", "sealed"])
    ap.add_argument("--families", default="", help="comma-separated subset to score")
    ap.add_argument("--sample", type=int, default=0, help="score a random N of the selection")
    ap.add_argument("--seed", type=int, default=20260813, help="which N --sample takes")
    ap.add_argument("--flip", type=int, default=0,
                    help="corrupt this many NVFP4 answers before scoring (failure demo)")
    ap.add_argument("--fragile-margin", type=float, default=-1.0,
                    help="BF16 top1-top2 margin below which an item's own answer is not "
                         "robust; default is the reference run's measured batching delta")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    doc = json.load(open(args.items))
    items = doc["items"]
    letter_ids = doc["letter_ids"]
    id2letter = {v: k for k, v in letter_ids.items()}
    refdoc = json.load(open(args.ref))
    ref = refdoc["results"]
    # How far the reference's own logits move when nothing about the model
    # changes -- measured by running one item alone and again inside the batch.
    # An item whose top-1 and top-2 are closer together than that has an answer
    # that is a property of the arithmetic's rounding, not of the model, and it
    # cannot carry weight in a comparison BETWEEN models.
    fragile_at = (args.fragile_margin if args.fragile_margin >= 0
                  else refdoc.get("selfcheck_max_abs_logit_delta", 0.0))

    if args.split != "all":
        items = [it for it in items if it.get("split", "open") == args.split]
    if args.families:
        want = set(args.families.split(","))
        items = [it for it in items if it["family"] in want]
    if args.sample:
        items = random.Random(args.seed).sample(items, min(args.sample, len(items)))
        items.sort(key=lambda it: it["key"])

    rows, missing = [], []
    for it in items:
        r = ref.get(it["key"])
        if r is not None:
            # The reference costs nine minutes a pass and is therefore the
            # artefact somebody will reuse after editing a prompt. Its cache is
            # keyed by the prompt's ids, and a stale entry is fatal rather than
            # a warning.
            want = hashlib.sha256(
                b"".join(int(i).to_bytes(8, "little") for i in it["ids"])).hexdigest()
            got = r.get("ids_sha256")
            if got is None:
                raise SystemExit(f"{args.ref} predates prompt-keyed caching; re-run the "
                                 f"reference so its entries can be checked against the items")
            if got != want:
                raise SystemExit(f"{it['key']}: the cached reference answers a different "
                                 f"prompt ({got[:12]} vs {want[:12]}); re-run the reference")
        f = read_fp4(os.path.join(args.fp4dir, it["key"]), it["ids"])
        if r is None or f is None:
            missing.append(it["key"])
            continue
        rows.append((it, r, f))

    if args.flip:
        # A synthetic regression of known size. The point is not to fake a
        # result, it is to find out what size of real regression this many
        # items can distinguish from nothing -- and to watch the numbers below
        # move when something is wrong, which is the only way to know they
        # would have.
        rng = random.Random(args.seed ^ 0x5EED)
        pick = rng.sample([i for i, (it, r, f) in enumerate(rows)
                           if r["argmax"] == it["answer_id"]], args.flip)
        for i in pick:
            it, r, f = rows[i]
            wrong = next(v for k, v in letter_ids.items() if v != it["answer_id"])
            f = dict(f)
            f["argmax"] = wrong
            f["flipped"] = True
            rows[i] = (it, r, f)
        print(f"!! --flip {args.flip}: {args.flip} NVFP4 answers deliberately corrupted "
              f"({', '.join(rows[i][0]['key'] for i in sorted(pick))})\n")

    n = len(rows)
    if missing:
        print(f"missing results for {len(missing)} items: {', '.join(missing[:8])}"
              f"{' ...' if len(missing) > 8 else ''}\n")
    if n == 0:
        raise SystemExit("nothing to score")

    ref_right = [r["argmax"] == it["answer_id"] for it, r, f in rows]
    fp4_right = [f["argmax"] == it["answer_id"] for it, r, f in rows]
    agree = [r["argmax"] == f["argmax"] for it, r, f in rows]
    ref_menu = [r["argmax"] in id2letter for it, r, f in rows]
    fp4_menu = [f["argmax"] in id2letter for it, r, f in rows]

    b = sum(1 for i in range(n) if ref_right[i] and not fp4_right[i])   # BF16 right, FP4 wrong
    c = sum(1 for i in range(n) if fp4_right[i] and not ref_right[i])   # the other way
    both = sum(1 for i in range(n) if ref_right[i] and fp4_right[i])
    neither = n - b - c - both

    print("=" * 78)
    print(f"PAIRED CAPABILITY: BF16 reference vs NVFP4 runtime   ({n} items, "
          f"split={args.split}{', families=' + args.families if args.families else ''})")
    print("=" * 78)
    print(f"  BF16 accuracy      {fmt_rate(sum(ref_right), n)}")
    print(f"  NVFP4 accuracy     {fmt_rate(sum(fp4_right), n)}")
    print(f"  chance             25.0%  (four options, key balanced 25/25/25/25)")
    print()
    print(f"  AGREEMENT          {fmt_rate(sum(agree), n)}   (same greedy token, no key needed)")
    print()
    print("  contingency (rows BF16, cols NVFP4)")
    print(f"      right/right {both:3d}   right/wrong {b:3d}")
    print(f"      wrong/right {c:3d}   wrong/wrong {neither:3d}")
    print()
    n_elig = both + b
    print(f"  LOSS   {fmt_rate(b, n_elig, 'of the items BF16 gets right, NVFP4 loses ')}")
    print(f"  GAIN   {c} item(s) NVFP4 gets right that BF16 does not")
    p = mcnemar_exact(b, c)
    print(f"  McNemar exact two-sided p = {p:.4f} on {b + c} discordant pair(s)"
          f"   {'-> a real difference' if p < 0.05 else '-> not distinguishable from noise'}")
    print()
    off_r, off_f = n - sum(ref_menu), n - sum(fp4_menu)
    print(f"  off-menu argmax    BF16 {off_r}/{n}, NVFP4 {off_f}/{n}   "
          f"(answered with a token that is not A/B/C/D at all)")

    margins = [r["top_logits"][0] - r["top_logits"][1] for it, r, f in rows]
    fragile = [i for i in range(n) if margins[i] < fragile_at]
    if fragile_at > 0:
        print()
        print(f"  reference wobble   {fragile_at:.3f} logits, measured (same item alone vs "
              f"in the batch)")
        print(f"  fragile items      {len(fragile)}/{n} have a BF16 top1-top2 margin below "
              f"that, so their own answer is rounding")
        if fragile:
            keep = [i for i in range(n) if i not in set(fragile)]
            kb = sum(1 for i in keep if ref_right[i] and not fp4_right[i])
            kc = sum(1 for i in keep if fp4_right[i] and not ref_right[i])
            ke = sum(1 for i in keep if ref_right[i])
            print(f"    dropping them:   AGREEMENT {sum(agree[i] for i in keep)}/{len(keep)}, "
                  f"LOSS {kb}/{ke}, GAIN {kc}, McNemar p = {mcnemar_exact(kb, kc):.4f}")
            print(f"    they are:        "
                  f"{', '.join(rows[i][0]['key'] + f' ({margins[i]:.2f})' for i in fragile)}")

    print()
    print("  per family")
    print(f"    {'family':<11} {'n':>3} {'BF16':>7} {'NVFP4':>7} {'agree':>7} {'lost':>6} {'gained':>7}")
    fams = sorted(set(it["family"] for it, r, f in rows))
    per_family = {}
    for fam in fams:
        ii = [i for i in range(n) if rows[i][0]["family"] == fam]
        fb = sum(1 for i in ii if ref_right[i] and not fp4_right[i])
        fc = sum(1 for i in ii if fp4_right[i] and not ref_right[i])
        per_family[fam] = {
            "n": len(ii),
            "bf16_right": sum(ref_right[i] for i in ii),
            "fp4_right": sum(fp4_right[i] for i in ii),
            "agree": sum(agree[i] for i in ii),
            "lost": fb, "gained": fc,
            "eligible": sum(1 for i in ii if ref_right[i]),
        }
        print(f"    {fam:<11} {len(ii):>3} {per_family[fam]['bf16_right']:>7} "
              f"{per_family[fam]['fp4_right']:>7} {per_family[fam]['agree']:>7} "
              f"{fb:>6} {fc:>7}")

    # Clustering, on two different populations: the losses (needs the key) and
    # the disagreements (does not).
    elig = [i for i in range(n) if ref_right[i]]
    p_loss, s_loss = cluster_test([rows[i][0]["family"] for i in elig],
                                  [not fp4_right[i] for i in elig])
    p_dis, s_dis = cluster_test([rows[i][0]["family"] for i in range(n)],
                                [not agree[i] for i in range(n)])
    print()
    print(f"  clustering of LOSSES across families    permutation p = {p_loss:.3f} "
          f"(chi2-like statistic {s_loss:.2f}, {sum(1 for i in elig if not fp4_right[i])} "
          f"loss(es) over {len(elig)} eligible)")
    print(f"  clustering of DISAGREEMENTS             permutation p = {p_dis:.3f} "
          f"(statistic {s_dis:.2f}, {n - sum(agree)} disagreement(s) over {n})")

    dis = [(it, r, f) for (it, r, f), a in zip(rows, agree) if not a]
    if dis:
        print()
        print("  every disagreement, in full")
        for it, r, f in dis:
            def show(ids, logits):
                return "  ".join(
                    f"{id2letter.get(i, '#' + str(i))}:{v:.2f}" for i, v in zip(ids, logits))
            mg = r["top_logits"][0] - r["top_logits"][1]
            print(f"    {it['key']:<16} {it['family']:<10} key={it['answer_letter']}"
                  f"   BF16 margin {mg:.2f}{'  (FRAGILE)' if mg < fragile_at else ''}")
            print(f"      BF16  -> {id2letter.get(r['argmax'], '#' + str(r['argmax']))}"
                  f"   {show(r['top_ids'][:5], r['top_logits'][:5])}")
            print(f"      NVFP4 -> {id2letter.get(f['argmax'], '#' + str(f['argmax']))}"
                  f"   {show(f['top_ids'][:5], f['top_logits'][:5])}")

    if args.json:
        json.dump({
            "n": n, "split": args.split, "families": args.families, "flip": args.flip,
            "bf16_right": sum(ref_right), "fp4_right": sum(fp4_right),
            "agree": sum(agree), "b_lost": b, "c_gained": c,
            "both": both, "neither": neither,
            "mcnemar_p": p, "cluster_p_loss": p_loss, "cluster_p_disagree": p_dis,
            "reference_wobble_logits": fragile_at,
            "fragile_items": [rows[i][0]["key"] for i in fragile],
            "per_family": per_family,
            "items": [{"key": it["key"], "family": it["family"],
                       "key_letter": it["answer_letter"],
                       "bf16": id2letter.get(r["argmax"], str(r["argmax"])),
                       "fp4": id2letter.get(f["argmax"], str(f["argmax"]))}
                      for it, r, f in rows],
        }, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
