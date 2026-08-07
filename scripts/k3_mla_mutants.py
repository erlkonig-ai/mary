#!/usr/bin/env python3
"""Mutation test for `k3_mla_gate`.

A gate that has never been seen to fail is decoration. For each claim the gate
makes, this makes a targeted wrong version of the thing being claimed, rebuilds,
runs the *real* gate, and records whether the gate caught it — then reverts.

A mutant that fails to COMPILE counts as caught (loudly, at the earliest
possible moment) and is recorded as such. A mutant the gate does not catch is
reported as SURVIVED, with no excuse attached.

Two mutants target the gate itself rather than the port: a comparator written
`d > tol` instead of `!(d <= tol)` (NaN-blind), and a no-op positive control.
Both are expected to let a broken thing through, which is the point — they show
what the disciplines in the gate are buying.

Usage:  python3 scripts/k3_mla_mutants.py [--only M01,M07] [--out mutants.json]
"""
import json
import os
import re
import subprocess
import sys
import time

# The checkout this harness lives in — correct in every clone.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLA = os.path.join(ROOT, "src/models/k3/mla.rs")
GATE = os.path.join(ROOT, "src/bin/k3_mla_gate.rs")
ENV = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"])

# (id, file, old, new, what it breaks, expectation)
#   expectation "caught"   -> the gate must fail (or the build must)
#   expectation "survives" -> documented: the mutant is not an error at all
MUTANTS = [
    ("M01", MLA,
     "        (self.q_head_dim() as f64).powf(-0.5)",
     "        (self.qk_nope_head_dim as f64).powf(-0.5)",
     "softmax scale 128^-0.5 instead of 192^-0.5 (forgetting the carried lane counts)",
     "caught"),
    ("M02", MLA,
     "pub const LORA_NORM_EPS: f64 = 1e-6;",
     "pub const LORA_NORM_EPS: f64 = 1e-5;",
     "MLA's internal RMSNorm epsilon set to config.rms_norm_eps instead of KimiRMSNorm's default",
     "caught"),
    ("M03", MLA,
     "    let normed = p.round(normed); // x.to(dtype)\n    p.round(normed * weight.clone().reshape([1, 1, d]))",
     "    p.round(normed * weight.clone().reshape([1, 1, d]))",
     "RMSNorm scales by the weight before casting back, rounding once instead of twice",
     "caught"),
    ("M04", MLA,
     "    let ms = (x.clone() * x.clone()).mean_dim(2); // [b, t, 1], fp32 island",
     "    let ms = (x.clone() * x.clone()).sum_dim(2); // [b, t, 1], fp32 island",
     "RMSNorm sums the squares instead of averaging them",
     "caught"),
    ("M05", MLA,
     "        let k_pass = kv4.clone().slice([0..b, 0..h, 0..t, 0..nope]);\n        let value = kv4.slice([0..b, 0..h, 0..t, nope..kvh]);",
     "        let k_pass = kv4.clone().slice([0..b, 0..h, 0..t, nope..kvh]);\n        let value = kv4.slice([0..b, 0..h, 0..t, 0..nope]);",
     "kv_b_proj's two halves swapped: value first, key second",
     "caught"),
    ("M06", MLA,
     "        kv_a_out.slice([0..b, 0..t, self.cfg.kv_lora_rank..w])",
     "        kv_a_out.slice([0..b, 0..t, 0..self.cfg.qk_carried_head_dim])",
     "carried key taken from kv_a_proj's head instead of its tail",
     "caught"),
    ("M07", MLA,
     "        (Tensor::cat(vec![pass, carried.clone()], 3), carried)",
     "        (Tensor::cat(vec![carried.clone(), pass], 3), carried)",
     "query assembled carried-first while the key stays passed-first (a one-sided permutation)",
     "caught"),
    ("M08", MLA,
     "        let q = q_b_out.reshape([b, t, h, qh]).swap_dims(1, 2);",
     "        let q = q_b_out.reshape([b, t, h, qh]).swap_dims(1, 2);\n        let q = {\n            let [bb, hh, tt, dd] = q.dims();\n            let head = q.clone().slice([0..bb, 0..hh, 0..tt, 0..nope]);\n            let tail = q.slice([0..bb, 0..hh, 0..tt, nope..dd]);\n            let half = (dd - nope) / 2;\n            let a = tail.clone().slice([0..bb, 0..hh, 0..tt, 0..half]);\n            let bq = tail.slice([0..bb, 0..hh, 0..tt, half..(dd - nope)]);\n            Tensor::cat(vec![head, bq.neg(), a], 3)\n        };",
     "THE HEADLINE MUTANT: a rotation applied to the carried lane of the query (rotate_half, the RoPE kernel)",
     "caught"),
    ("M09", MLA,
     "                p.round(flat * p.round(sigmoid(g.clone())))",
     "                p.round(flat)",
     "the sigmoid output gate dropped entirely",
     "caught"),
    ("M10", MLA,
     "                p.round(flat * p.round(sigmoid(g.clone())))",
     "                p.round(flat * p.round(g.clone()))",
     "the output gate applied without its sigmoid",
     "caught"),
    ("M11", MLA,
     "        let k_carried = kv_carried.reshape([b, 1, t, carried]).repeat_dim(1, h);",
     "        let k_carried = kv_carried\n            .slice([0..b, 0..1, 0..carried])\n            .reshape([b, 1, 1, carried])\n            .repeat_dim(2, t)\n            .repeat_dim(1, h);",
     "carried key broadcast from token 0 to every token, not per token",
     "caught"),
    ("M12", MLA,
     "                    v.push(if j <= offset + i { 0.0f64 } else { neg });",
     "                    v.push(if j <= offset + i + 1 { 0.0f64 } else { neg });",
     "causal mask off by one: every token may attend one step into the future",
     "caught"),
    ("M13", MLA,
     "                    v.push(if j <= offset + i { 0.0f64 } else { neg });",
     "                    v.push(0.0f64 * (j + offset + i) as f64);",
     "no causal mask at all: full bidirectional attention",
     "caught"),
    ("M14", MLA,
     "        let precast = softmax_dim(scores, 3);",
     "        let precast = softmax_dim(scores, 2);",
     "softmax over the query axis instead of the key axis",
     "caught"),
    ("M15", MLA,
     "        self.precision.round(probs.matmul(v)).swap_dims(1, 2)",
     "        self.precision.round(probs.matmul(v))",
     "probs*v left in [B,H,T,dv] so the flatten interleaves heads and tokens",
     "caught"),
    ("M16", MLA,
     "        let cast = self.precision.round(precast.clone());",
     "        let cast = precast.clone();",
     "the softmax's cast back to the activation dtype skipped (fp32 island left open)",
     "caught"),
    ("M17", MLA,
     "        let kv_a_layernorm_out = self.kv_a_norm(kv_a_layernorm_in.clone());",
     "        let kv_a_layernorm_out = kv_a_layernorm_in.clone();",
     "kv_a_layernorm skipped: the raw latent fed to kv_b_proj",
     "caught"),
    ("M18", MLA,
     "            Some(prev) => Tensor::cat(vec![prev, k], 2),",
     "            Some(prev) => Tensor::cat(vec![k, prev], 2),",
     "KV cache appends new tokens BEFORE the history",
     "caught"),
    ("M19", GATE,
     '        let name = format!("language_model.model.layers.{}.self_attn.{}.weight", layer, part);',
     '        let name = format!(\n            "language_model.model.layers.{}.self_attn.{}.weight",\n            if *part == "q_b_proj" { layer + 4 } else { layer },\n            part\n        );',
     "q_b_proj loaded from the next MLA layer's shard (a plausible weight-map slip)",
     "caught"),
    ("M20", GATE,
     '    let (o, os) = take("o_proj");',
     '    let (o, os) = take("o_proj");\n    let o: Vec<f64> = o.into_iter().map(|x| -x).collect();',
     "o_proj weights negated on load (the shipped-negated-weights failure mode)",
     "caught"),
    # ---- gate-side mutants: what the gate's own disciplines are buying ----
    ("G01", MLA,
     "                let v: Vec<f64> = v.into_iter().map(|x| bf16::from_f64(x).to_f64()).collect();",
     "                let v: Vec<f64> = v.into_iter().enumerate().map(|(i, x)| if i == 7 { f64::NAN } else { bf16::from_f64(x).to_f64() }).collect();",
     "the port emits a single NaN element in every bfloat16 tensor — does the gate notice, or does a NaN score as zero error?",
     "caught"),
    ("G02", GATE,
     "fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {\n    let mut m = 0.0f64;\n    for (x, y) in a.iter().zip(b) {\n        let d = (x - y).abs();\n        if d.is_nan() {\n            return f64::NAN;\n        }\n        if d > m {\n            m = d;\n        }\n    }\n    m\n}",
     "fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {\n    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))\n}",
     "GATE MUTANT: the NaN-propagating max written the obvious `fold(0.0, f64::max)` way, with G01's NaN in flight and the bit-agreement floor removed",
     "survives"),
    ("G03", GATE,
     '    run_subblock_lane(&mut g, "f32", 3, &lane32, &blk32, &dev32, Tol::Rel(1e-4), None);',
     '    // lane deliberately skipped',
     "GATE MUTANT: a whole lane silently removed — does the check count notice?",
     "caught"),
    ("G04", GATE,
     "                    let ang = ti as f64 * inv;",
     "                    let ang = 0.0 * ti as f64 * inv;",
     "GATE MUTANT: the RoPE positive control made a no-op — the control must fail",
     "caught"),
    ("G05", GATE,
     'const SHA_PREFIX13: &str = "fdb3b897f0bb43e8506d27dd283defee87910006dd1038c131687a1b48e61d7c";',
     'const SHA_PREFIX13: &str = "0db3b897f0bb43e8506d27dd283defee87910006dd1038c131687a1b48e61d7c";',
     "GATE MUTANT: the oracle is not the pinned oracle",
     "caught"),
    ("G06", GATE,
     '    g.cmp_exact("tf", "assemble_query", &format!("{} -> {}", qb_key, okey), &host(&q_states), &gold);',
     '    let q_states = q_states.mul_scalar(1.0000001f64);\n        g.cmp_exact("tf", "assemble_query", &format!("{} -> {}", qb_key, okey), &host(&q_states), &gold);',
     "a 1e-7 perturbation of an assembly that must be bit-exact",
     "caught"),
]


def run(cmd, cwd=ROOT, timeout=2400):
    return subprocess.run(cmd, cwd=cwd, env=ENV, shell=True, capture_output=True,
                          text=True, timeout=timeout)


def patch(path, old, new):
    src = open(path).read()
    if src.count(old) != 1:
        raise SystemExit("mutant anchor appears %d times in %s:\n%s"
                         % (src.count(old), path, old[:200]))
    open(path, "w").write(src.replace(old, new, 1))


def build_and_run(tag):
    b = run("cargo build --release --features k3-mla --bin k3_mla_gate")
    if b.returncode != 0:
        err = "\n".join(l for l in b.stderr.splitlines() if l.startswith("error"))[:400]
        return {"build": "FAILED", "gate": "not reached", "detail": err}
    g = run("./target/release/k3_mla_gate --json /tmp/mut_%s.json" % tag)
    out = g.stdout + g.stderr
    if "panicked at" in out:
        line = [l for l in out.splitlines() if "panicked at" in l or
                (l.strip() and "assertion" in l)][:2]
        return {"build": "ok", "gate": "PANIC", "detail": " | ".join(line)[:400]}
    m = re.search(r"(\d+) checks, (\d+) failed", out)
    nfail = int(m.group(2)) if m else -1
    names = [l.split()[1] for l in out.splitlines() if l.startswith("FAIL ")][:6]
    return {"build": "ok",
            "gate": "PASS" if g.returncode == 0 else "FAIL",
            "n_failed": nfail,
            "failing": names,
            "detail": ""}


def main():
    only = None
    out_path = os.path.join(ROOT, "golden/k3_mla_mutants.json")
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--only":
            only = set(args[i + 1].split(","))
        if a == "--out":
            out_path = args[i + 1]

    originals = {MLA: open(MLA).read(), GATE: open(GATE).read()}
    results = []
    t0 = time.time()

    print("== baseline: the unmutated gate must pass ==", flush=True)
    base = build_and_run("base")
    print("   ", base, flush=True)
    if base["gate"] != "PASS":
        raise SystemExit("baseline gate does not pass; mutation testing is meaningless")

    try:
        for mid, path, old, new, what, expect in MUTANTS:
            if only and mid not in only:
                continue
            print("== %s %s" % (mid, what), flush=True)
            patch(path, old, new)
            # G02 only means anything with a NaN in flight, so it rides G01.
            if mid == "G02":
                # G02 is a *gate* mutant: it only means anything with something
                # broken in flight, so it carries G01's NaN, and it also drops
                # the bit-agreement floor so that the numeric comparator is the
                # only thing left standing. The point is to show that both
                # disciplines are load bearing, not just one.
                patch(MLA,
                      "                let v: Vec<f64> = v.into_iter().map(|x| bf16::from_f64(x).to_f64()).collect();",
                      "                let v: Vec<f64> = v.into_iter().enumerate().map(|(i, x)| if i == 7 { f64::NAN } else { bf16::from_f64(x).to_f64() }).collect();")
                patch(GATE, "    const MIN_BITEXACT: f64 = 0.99;", "    const MIN_BITEXACT: f64 = 0.0;")
                patch(GATE,
                      "        let ok = !saw_nan\n            && mask_mismatch == 0",
                      "        let ok = mask_mismatch == 0")
            r = build_and_run(mid)
            for f, src in originals.items():
                open(f, "w").write(src)
            caught = r["gate"] in ("FAIL", "PANIC") or r["build"] == "FAILED"
            r.update(id=mid, file=os.path.basename(path), mutation=what,
                     expectation=expect, caught=caught,
                     verdict="CAUGHT" if caught else "SURVIVED",
                     as_expected=(caught == (expect == "caught")))
            print("   ", r["verdict"], r.get("failing", ""), r["detail"][:160],
                  "(%.0fs)" % (time.time() - t0), flush=True)
            results.append(r)
    finally:
        for f, src in originals.items():
            open(f, "w").write(src)

    print("== restoring and re-running the clean gate ==", flush=True)
    final = build_and_run("final")
    print("   ", final, flush=True)

    n = len(results)
    caught = sum(1 for r in results if r["caught"])
    unexpected = [r["id"] for r in results if not r["as_expected"]]
    summary = {
        "baseline": base, "final": final, "n": n, "caught": caught,
        "survived": n - caught, "unexpected": unexpected, "mutants": results,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(summary, open(out_path, "w"), indent=1)
    print("\n%d mutants: %d caught, %d survived (%s expected to survive). unexpected: %s"
          % (n, caught, n - caught,
             sum(1 for _, _, _, _, _, e in MUTANTS if e == "survives"),
             unexpected or "none"), flush=True)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
