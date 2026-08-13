#!/usr/bin/env python3
"""What about the tokens after the first? Teacher-forced divergence.

The multiple-choice half of this harness measures one token, because a second
generated token from the BF16 reference costs another whole pass over 531.9 GB.
That leaves a real hole: a quantisation fault that only shows up after fifty
tokens of prose is invisible to it.

This closes most of the hole for the price of ONE reference stream. The NVFP4
runtime generates a continuation on its own (`INK_GEN`, KV cache on). Then the
prompt AND that continuation go through the reference as one sequence, and the
reference is asked what IT would have emitted at every position. Every position
of the continuation is scored by the same forward, because a forward already
computes a hidden state at each of them.

Two numbers come out:

  * per-token agreement — over all generated positions, how often the reference
    would have emitted the token the runtime actually did;
  * first divergence — the index of the first position where it would not,
    which is the length of continuation you can trust to be identical.

Say what this is NOT. It is TEACHER-FORCED: the reference is conditioned on the
runtime's tokens, so it answers "would you have said this, here", not "do you
generate the same text". Those come apart after the first divergence, where free
generation would wander off and teacher forcing stays on the runtime's path.
Teacher forcing is the conservative one and the affordable one: free generation
from the reference is one full stream per token.

  gen_divergence.py build   <items.json> <fp4gendir> <out.json>
  gen_divergence.py report  <out.json> <ref.json>
  gen_divergence.py compare <out.json> <ref.json> <fp4_tf_dir>

`build` reads the NVFP4 generation logs and writes reference items carrying
`score_from`; feed that to `inkling_bf16_stream.py`; then `report`.
"""
import json
import os
import re
import sys

# `paired_score` is the sibling module, and this is spelled out because these
# scripts are run with `python -P` (the interpreter that does NOT put the
# script's own directory on the path) after a stray `/tmp/h2.py` in a shared
# scratch directory shadowed `h2` and broke every transformers import on the
# box. Safe path, explicit import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STEP = re.compile(r"^\s*step \d+: \+(\d+)", re.M)


def build(items_path, gendir, out_path):
    items = json.load(open(items_path))["items"]
    out = []
    for it in items:
        log = os.path.join(gendir, it["key"], "tail.log")
        if not os.path.exists(log):
            continue
        gen = [int(m.group(1)) for m in STEP.finditer(open(log, errors="replace").read())]
        if not gen:
            continue
        # The last step emits a token that is never fed back, so the reference
        # has no hidden state conditioned on it; drop it rather than score a
        # position that does not exist.
        gen = gen[:-1] if len(gen) > 1 else gen
        ids = list(it["ids"]) + gen
        out.append({
            "key": it["key"],
            "ids": ids,
            # Predictions start at the last PROMPT position, which is the one
            # whose argmax is the first generated token.
            "score_from": len(it["ids"]) - 1,
            "generated": gen,
            "n_prompt": len(it["ids"]),
            "family": it.get("family", ""),
        })
    json.dump({"items": out}, open(out_path, "w"), indent=1)
    print(f"{out_path}: {len(out)} sequences, "
          f"{sum(len(o['generated']) for o in out)} generated tokens to score, "
          f"max length {max((len(o['ids']) for o in out), default=0)}")


def report(built_path, ref_path):
    built = {it["key"]: it for it in json.load(open(built_path))["items"]}
    ref = json.load(open(ref_path))["results"]
    tot = agree = 0
    print(f"{'item':<16} {'gen':>4} {'agree':>7} {'first divergence':>18}")
    rows = []
    for key, it in built.items():
        r = ref.get(key)
        if r is None or "positions" not in r:
            continue
        pred = r["positions"]["argmax"]
        gen = it["generated"]
        n = min(len(pred), len(gen))
        same = [pred[i] == gen[i] for i in range(n)]
        first = next((i for i, ok in enumerate(same) if not ok), None)
        tot += n
        agree += sum(same)
        rows.append((key, n, sum(same), first))
        print(f"{key:<16} {n:>4} {sum(same):>3}/{n:<3} "
              f"{('at token ' + str(first)) if first is not None else 'none':>18}")
    if tot == 0:
        raise SystemExit("nothing scored")
    from paired_score import wilson
    lo, hi = wilson(agree, tot)
    print()
    print(f"teacher-forced per-token agreement: {agree}/{tot} = {100 * agree / tot:.1f}% "
          f"[95% CI {100 * lo:.1f}–{100 * hi:.1f}]")
    firsts = [f for _, _, _, f in rows if f is not None]
    n_clean = sum(1 for _, _, _, f in rows if f is None)
    print(f"sequences identical all the way through: {n_clean}/{len(rows)}")
    if firsts:
        firsts.sort()
        print(f"first divergence, when it happens: median token "
              f"{firsts[len(firsts) // 2]}, range {firsts[0]}..{firsts[-1]}")


def compare(built_path, ref_path, tfdir):
    """The apples-to-apples version: both models teacher-forced on the same ids.

    `report` compares the reference's teacher-forced argmax against the tokens
    the runtime EMITTED, which came out of the KV-cached decode lane. Those are
    two different lanes as well as two different models, and the comparison
    cannot tell the two apart. Running the same prompt-plus-continuation through
    the runtime as a plain uncached forward puts both sides in the same lane on
    the same input, so what is left is the model difference.

    It also gives the cached-versus-uncached number for free, by comparing the
    runtime's own two lanes on the same sequence -- which is a property of our
    implementation and nothing to do with BF16.
    """
    built = {it["key"]: it for it in json.load(open(built_path))["items"]}
    ref = json.load(open(ref_path))["results"]
    LINE = re.compile(r"after token (\d+) \(id \d+\): top5 \[(\d+)")
    tot = agree = 0
    lane_tot = lane_agree = 0
    print(f"{'item':<16} {'n':>4} {'BF16 vs NVFP4':>15} {'NVFP4 cached vs not':>21}")
    for key, it in built.items():
        r = ref.get(key)
        log = os.path.join(tfdir, key, "tail.log")
        if r is None or "positions" not in r or not os.path.exists(log):
            continue
        fp = {}
        for m in LINE.finditer(open(log, errors="replace").read()):
            fp[int(m.group(1))] = int(m.group(2))
        k = r["positions"]["score_from"]
        rp = r["positions"]["argmax"]
        gen = it["generated"]
        n = min(len(rp), len(gen), sum(1 for i in range(len(rp)) if k + i in fp))
        same = [rp[i] == fp[k + i] for i in range(n)]
        lane = [fp[k + i] == gen[i] for i in range(n)]
        tot += n
        agree += sum(same)
        lane_tot += n
        lane_agree += sum(lane)
        print(f"{key:<16} {n:>4} {sum(same):>7}/{n:<7} {sum(lane):>13}/{n:<7}")
    if tot == 0:
        raise SystemExit("nothing to compare")
    from paired_score import wilson
    lo, hi = wilson(agree, tot)
    print()
    print(f"BF16 vs NVFP4, both teacher-forced on the same ids: {agree}/{tot} = "
          f"{100 * agree / tot:.1f}% [95% CI {100 * lo:.1f}–{100 * hi:.1f}]")
    llo, lhi = wilson(lane_agree, lane_tot)
    print(f"NVFP4 uncached vs its own KV-cached generation:     {lane_agree}/{lane_tot} = "
          f"{100 * lane_agree / lane_tot:.1f}% [95% CI {100 * llo:.1f}–{100 * lhi:.1f}]")
    print()
    print("The second line is OUR two lanes on one model and has nothing to do with BF16.")
    print("Whatever it costs is a floor under the first line, in the same way the")
    print("run-to-run flip rate is a floor under the multiple-choice result.")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "report":
        report(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
