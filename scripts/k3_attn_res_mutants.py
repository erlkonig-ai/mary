#!/usr/bin/env python3
"""Mutation-test k3_attn_res_gate.

For each mutant: apply an exact textual edit to the PORT (never to the gate),
rebuild, run the gate, record whether it failed, then restore the file. A gate
that never fails is decoration; this is the evidence that it can.
"""
import json
import os
import subprocess
import sys

ROOT = "<worktree>"
FEATURES = os.environ.get("K3_GATE_FEATURES", "k3-attn-res")
AR = os.path.join(ROOT, "src/models/k3/attn_res.rs")
CF = os.path.join(ROOT, "src/models/k3/config.rs")

# (id, description, [(file, old, new), ...])
MUTANTS = [
    ("M01", "candidate stack: accumulator FIRST instead of last", [
        (AR, "    slots.push(accumulator.reshape([tokens, 1, hidden]));\n    Tensor::cat(slots, 1)",
             "    slots.insert(0, accumulator.reshape([tokens, 1, hidden]));\n    Tensor::cat(slots, 1)")]),
    ("M02", "snapshot the POST-MIX hidden state, not the raw layer input", [
        (AR, "            self.bank.push(layer_in);\n            self.accumulator = None;",
             "            self.bank.push(to_attention.clone());\n            self.accumulator = None;")]),
    ("M03", "no reset: the accumulator survives the boundary", [
        (AR, "            self.bank.push(layer_in);\n            self.accumulator = None;",
             "            self.bank.push(layer_in.clone());\n            self.accumulator = Some(layer_in);")]),
    ("M04", "reset to the snapshot instead of to nothing", [
        (AR, "            None => attn_out,\n        };",
             "            None => round_bf16(self.bank[self.bank.len() - 1].clone() + attn_out),\n        };")]),
    ("M05", "boundary off-by-one: (layer + 1) % period == 0", [
        (CF, ".is_some_and(|period| layer % period == 0)",
             ".is_some_and(|period| (layer + 1) % period == 0)")]),
    ("M06", "boundary off-by-one the other way: (layer + period - 1) % period == 0", [
        (CF, ".is_some_and(|period| layer % period == 0)",
             ".is_some_and(|period| (layer + period - 1) % period == 0)")]),
    ("M07", "wrong block size: period + 1", [
        (CF, ".is_some_and(|period| layer % period == 0)",
             ".is_some_and(|period| layer % (period + 1) == 0)")]),
    ("M08", "block size 8 instead of the config's", [
        (CF, ".is_some_and(|period| layer % period == 0)",
             ".is_some_and(|_period| layer % 8 == 0)")]),
    ("M09", "mix over the NORMALISED k instead of the raw v", [
        (AR, "        let scores = tree_sum_last(k * self.score_weight.clone().reshape([1, 1, hidden]))",
             "        let scores = tree_sum_last(k.clone() * self.score_weight.clone().reshape([1, 1, hidden]))"),
        (AR, "        let out = (probs.clone().reshape([tokens, slots, 1]) * v)",
             "        let out = (probs.clone().reshape([tokens, slots, 1]) * k)")]),
    ("M10", "scores from the RAW v instead of the normalised k", [
        (AR, "        let scores = tree_sum_last(k * self.score_weight.clone().reshape([1, 1, hidden]))",
             "        let _ = k;\n        let scores = tree_sum_last(v.clone() * self.score_weight.clone().reshape([1, 1, hidden]))")]),
    ("M11", "epsilon dropped from the RMSNorm denominator", [
        (AR, "let k = v.clone() / variance.add_scalar(self.eps).sqrt();",
             "let k = v.clone() / variance.sqrt();")]),
    ("M12", "epsilon added OUTSIDE the sqrt", [
        (AR, "let k = v.clone() / variance.add_scalar(self.eps).sqrt();",
             "let k = v.clone() / variance.sqrt().add_scalar(self.eps);")]),
    ("M13", "the reciprocal that started all this: sqrt().recip()", [
        (AR, "let k = v.clone() / variance.add_scalar(self.eps).sqrt();",
             "let k = v.clone() * variance.add_scalar(self.eps).sqrt().recip();")]),
    ("M14", "flat reduction instead of pairwise (both reductions)", [
        (AR, "        let variance = tree_sum_last(v.clone().powf_scalar(2.0)).div_scalar(hidden as f64);",
             "        let variance = v.clone().powf_scalar(2.0).sum_dim(2).div_scalar(hidden as f64);"),
        (AR, "        let scores = tree_sum_last(k * self.score_weight.clone().reshape([1, 1, hidden]))",
             "        let scores = (k * self.score_weight.clone().reshape([1, 1, hidden])).sum_dim(2)")]),
    ("M15", "variance not divided by the width", [
        (AR, ".div_scalar(hidden as f64);\n        let k =", ".div_scalar(1.0f64);\n        let k =")]),
    ("M16", "softmax over the token axis instead of the slot axis", [
        (AR, "let probs = softmax(scores.clone(), 1);", "let probs = softmax(scores.clone(), 0);")]),
    ("M17", "mixture output not rounded to bfloat16", [
        (AR, "AttnResMix { scores, probs, out: round_bf16(out) }",
             "AttnResMix { scores, probs, out }")]),
    ("M18", "accumulator add not rounded to bfloat16 (after attention)", [
        (AR, "            Some(prefix) => round_bf16(prefix + attn_out),",
             "            Some(prefix) => prefix + attn_out,")]),
    ("M19", "accumulator add not rounded to bfloat16 (after MLP)", [
        (AR, "        let accumulator = round_bf16(\n            self.accumulator.take().expect(\"accumulator missing after attention\") + mlp_out,\n        );",
             "        let accumulator =\n            self.accumulator.take().expect(\"accumulator missing after attention\") + mlp_out;")]),
    ("M20", "score weight = the norm gain alone (projection dropped)", [
        (AR, "            score_weight: norm_weight * proj_weight.reshape([hidden]),",
             "            score_weight: { let _ = proj_weight; norm_weight },")]),
    ("M21", "score weight = the projection alone (norm gain dropped)", [
        (AR, "            score_weight: norm_weight * proj_weight.reshape([hidden]),",
             "            score_weight: { let _ = norm_weight; proj_weight.reshape([hidden]) },")]),
    ("M22", "score weight summed instead of multiplied", [
        (AR, "            score_weight: norm_weight * proj_weight.reshape([hidden]),",
             "            score_weight: norm_weight + proj_weight.reshape([hidden]),")]),
    ("M23", "the self-attention mixture runs even with an empty bank", [
        (AR, "        let mix = if self.bank.is_empty() {\n            None\n        } else {",
             "        let mix = if false {\n            None\n        } else {")]),
    ("M24", "snapshot taken BEFORE the entry mixture reads the bank", [
        (AR, "        let mix = if self.bank.is_empty() {\n            None\n        } else {\n            Some(sa.mix(stack_candidates(&self.bank, layer_in.clone())))\n        };",
             "        if self.schedule[self.layer] {\n            self.bank.push(layer_in.clone());\n        }\n        let mix = if self.bank.is_empty() {\n            None\n        } else {\n            Some(sa.mix(stack_candidates(&self.bank, layer_in.clone())))\n        };")]),
    ("M25", "accumulator seeded with the MIXED value, not the raw layer input", [
        (AR, "            self.accumulator = Some(layer_in);\n        }",
             "            self.accumulator = Some(to_attention.clone());\n        }")]),
    ("M26", "layer output is the MLP output alone (residual add dropped)", [
        (AR, "        let accumulator = round_bf16(\n            self.accumulator.take().expect(\"accumulator missing after attention\") + mlp_out,\n        );",
             "        let _ = self.accumulator.take();\n        let accumulator = round_bf16(mlp_out);")]),
    ("M27", "round_bf16 splitting constant 65536 instead of 65537", [
        (AR, "const DEKKER_C: f32 = 65537.0;", "const DEKKER_C: f32 = 65536.0;")]),
    ("M28", "tree reduction drops the odd tail", [
        (AR, "        if n % 2 == 1 {\n            next = Tensor::cat(vec![next, cur.narrow(2, n - 1, 1)], 2);\n            n = half + 1;\n        } else {\n            n = half;\n        }",
             "        n = half;")]),
    ("M29", "call-order guard removed", [
        (AR, "        assert_eq!(self.stage, Stage::Entry, \"enter_layer out of order at layer {}\", self.layer);",
             "        self.stage = Stage::Entry;")]),
    ("M30", "projection rank assertion removed", [
        (AR, "        assert_eq!(\n            rows, 1,",
             "        assert_eq!(\n            rows, rows,")]),
    ("M31", "finish() mixes over the bank WITHOUT the final hidden state", [
        (AR, "        out.mix(stack_candidates(&self.bank, hidden))",
             "        out.mix(stack_candidates(&self.bank[..self.bank.len() - 1], hidden))")]),
    ("M35", "mixture via matmul instead of broadcast-multiply-and-sum "
            "(CPU-invisible; only the CUDA lane can catch this)", [
        (AR, "        let out = (probs.clone().reshape([tokens, slots, 1]) * v)\n            .sum_dim(1)\n            .reshape([tokens, hidden]);",
             "        let out = probs.clone().reshape([tokens, 1, slots]).matmul(v).reshape([tokens, hidden]);")]),
    ("M33", "round_bf16 is the identity (no rounding at all)", [
        (AR, "    let c = x.clone().mul_scalar(DEKKER_C);\n    c.clone() - (c - x)",
             "    let _ = DEKKER_C;\n    x")]),
    ("M34", "bank entries stacked newest-first", [
        (AR, "    for entry in bank {", "    for entry in bank.iter().rev() {")]),
    ("M32", "layer 0's schedule flag forced off", [
        (CF, ".is_some_and(|period| layer % period == 0)",
             ".is_some_and(|period| layer != 0 && layer % period == 0)")]),
]


def read(p):
    with open(p) as f:
        return f.read()


def write(p, s):
    with open(p, "w") as f:
        f.write(s)


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True, text=True)


def main():
    only = sys.argv[1:] or None
    originals = {AR: read(AR), CF: read(CF)}
    results = []
    try:
        for mid, desc, edits in MUTANTS:
            if only and mid not in only:
                continue
            # apply
            bad = None
            for path, old, new in edits:
                s = read(path)
                if old not in s:
                    bad = f"anchor not found in {os.path.basename(path)}"
                    break
                if s.count(old) != 1:
                    bad = f"anchor is not unique in {os.path.basename(path)} ({s.count(old)}x)"
                    break
                write(path, s.replace(old, new, 1))
            if bad:
                for p, s in originals.items():
                    write(p, s)
                results.append({"id": mid, "desc": desc, "outcome": "BROKEN-MUTANT", "detail": bad})
                print(f"{mid} {desc}\n    !! {bad}", flush=True)
                continue

            b = run(f"PATH=$HOME/.cargo/bin:$PATH cargo build --release --features {FEATURES} "
                    f"--bin k3_attn_res_gate 2>&1")
            if b.returncode != 0:
                tail = [l for l in b.stdout.splitlines() if l.startswith("error")][:2]
                outcome = "DID-NOT-COMPILE"
                detail = " | ".join(tail)
                failures = []
            else:
                r = run("./target/release/k3_attn_res_gate 2>/dev/null")
                failures = [l.strip()[2:] for l in r.stdout.splitlines()
                            if l.strip().startswith("- ")]
                caught = r.returncode != 0
                outcome = "CAUGHT" if caught else "SURVIVED"
                detail = "; ".join(f.split(":")[0] for f in failures[:6]) or "gate reported no failure"
                if "GATE: PASS" not in r.stdout and "GATE: FAIL" not in r.stdout:
                    outcome = "CAUGHT" if caught else "SURVIVED"
                    detail = "gate aborted: " + (r.stdout.strip().splitlines() or ["<no output>"])[-1][:160]
            results.append({"id": mid, "desc": desc, "outcome": outcome, "detail": detail,
                            "n_failed_checks": len(failures)})
            print(f"{mid} {outcome:<16} {desc}\n    {detail}", flush=True)

            for p, s in originals.items():
                write(p, s)
    finally:
        for p, s in originals.items():
            write(p, s)

    with open("<path>", "w") as f:
        json.dump(results, f, indent=1)
    n = len(results)
    caught = sum(1 for r in results if r["outcome"] == "CAUGHT")
    print(f"\n{caught}/{n} caught")
    for r in results:
        if r["outcome"] != "CAUGHT":
            print(f"  {r['id']} {r['outcome']}: {r['desc']} — {r['detail']}")


if __name__ == "__main__":
    main()
