#!/usr/bin/env python3
"""Mutation-test the Kimi K3 MoE router gate.

For each mutant: apply an exact textual edit (or a corrupted artifact), rebuild,
run the gate, record whether it FAILED, revert. A gate that has never been seen
to fail is decoration.

Every source edit asserts its anchor occurs exactly once, so a silently-missed
mutation cannot masquerade as a survivor.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import time

# The checkout this harness lives in — correct in every clone, unlike a default.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTER = os.path.join(REPO, "src/models/k3/router.rs")
GATE = os.path.join(REPO, "src/bin/k3_router_gate.rs")
VEC = os.environ.get("K3_ORACLE_DIR", "./k3-oracle")
WORK = os.environ.get("K3_WORK_DIR", "./k3-work")
ENV = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"])

PRISTINE = {ROUTER: io.open(ROUTER, encoding="utf-8").read(),
            GATE: io.open(GATE, encoding="utf-8").read()}


def restore():
    for p, s in PRISTINE.items():
        io.open(p, "w", encoding="utf-8").write(s)


def apply_edits(edits):
    """edits: list of (path, old, new). Each `old` must occur exactly once."""
    bufs = {p: s for p, s in PRISTINE.items()}
    for path, old, new in edits:
        n = bufs[path].count(old)
        if n != 1:
            raise SystemExit(f"ANCHOR NOT UNIQUE ({n}) in {path}:\n{old[:200]}")
        bufs[path] = bufs[path].replace(old, new)
    for p, s in bufs.items():
        io.open(p, "w", encoding="utf-8").write(s)


def build():
    r = subprocess.run(
        ["cargo", "build", "--release", "--features", "kimi-k3", "--bin", "k3_router_gate"],
        cwd=REPO, env=ENV, capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-2000:]


def run_gate(vecdir=VEC):
    r = subprocess.run([os.path.join(REPO, "target/release/k3_router_gate"), vecdir],
                       cwd=REPO, env=ENV, capture_output=True, text=True, timeout=1800)
    return r.returncode, r.stdout, r.stderr


def failures(stdout):
    lines = [l for l in stdout.splitlines() if l.rstrip().endswith("FAIL")]
    return lines


# ---------------------------------------------------------------------------
# corrupted-artifact mutants: build a vectors dir with a tampered weight npz
# ---------------------------------------------------------------------------

def make_mut_vecdir(tag, transform):
    """transform(arrays_dict, manifest_dict) -> (arrays, manifest)"""
    import numpy as np
    d = os.path.join(WORK, "mutvec_" + tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    os.symlink(os.path.join(VEC, "layer_oracle_prefix13_bf16.npz"),
               os.path.join(d, "layer_oracle_prefix13_bf16.npz"))
    z = np.load(os.path.join(VEC, "k3router_gateweights_routerport.npz"))
    arrays = {k: z[k] for k in z.files}
    manifest = json.load(open(os.path.join(VEC, "k3router_gateweights_routerport_manifest.json")))
    arrays, manifest = transform(arrays, manifest)
    np.savez(os.path.join(d, "k3router_gateweights_routerport.npz"), **arrays)
    json.dump(manifest, open(os.path.join(d, "k3router_gateweights_routerport_manifest.json"), "w"))
    return d


def t_negate(a, m):
    import numpy as np
    # flip the bf16 sign bit of every gate weight in layer 5 — the exact
    # "shipped negated weights" failure the brief calls out
    a["L05_gate_weight_bf16bits"] = (a["L05_gate_weight_bf16bits"] ^ np.uint16(0x8000))
    return a, m


def t_truncate(a, m):
    import numpy as np
    a["L03_gate_bias_f32"] = np.zeros((0,), dtype=np.float32)
    m["arrays"]["L03_gate_bias_f32"]["shape"] = [0]
    import hashlib
    m["arrays"]["L03_gate_bias_f32"]["sha256"] = hashlib.sha256(b"").hexdigest()
    return a, m


def t_manifest_lie(a, m):
    m["arrays"]["L07_gate_weight_bf16bits"]["sha256"] = "0" * 64
    return a, m


def t_bias_f32_is_bf16(a, m):
    # make the f32 lane identical to the bf16 lane: the pinned divergence count
    # (6 of 384) must move
    import hashlib
    import numpy as np
    b = a["L12_gate_bias_bf16bits"].astype(np.uint32) << 16
    a["L12_gate_bias_f32"] = b.view(np.float32).astype(np.float32)
    m["arrays"]["L12_gate_bias_f32"]["sha256"] = hashlib.sha256(
        a["L12_gate_bias_f32"].tobytes()).hexdigest()
    return a, m


def t_bias_perturb(a, m):
    import hashlib
    import numpy as np
    b = a["L09_gate_bias_bf16bits"].copy()
    b[17] = np.uint16(int(b[17]) ^ 1)  # one ulp
    a["L09_gate_bias_bf16bits"] = b
    m["arrays"]["L09_gate_bias_bf16bits"]["sha256"] = hashlib.sha256(b.tobytes()).hexdigest()
    return a, m


def t_drop_array(a, m):
    del a["L06_gate_weight_bf16bits"]
    del m["arrays"]["L06_gate_weight_bf16bits"]
    return a, m


# ---------------------------------------------------------------------------
# the mutants
# ---------------------------------------------------------------------------

CFG_BLOCK = """            hidden_size: 7168,
            num_experts: 896,
            top_k: 16,
            activation: RouterActivation::Sigmoid,
            renormalize: true,
            routed_scaling_factor: 1.0,
            num_expert_group: 1,
            topk_group: 1,"""

LOGITS_TAIL = """                };
            }
        }
        out
    }"""

NORM_TAIL = """        if self.cfg.routed_scaling_factor != 1.0 {
            for x in out.iter_mut() {
                *x *= self.cfg.routed_scaling_factor;
            }
        }
        out
    }"""

NAN_EDIT = (ROUTER, NORM_TAIL,
            NORM_TAIL.replace("        out\n    }",
                              "        out[0] = f32::NAN;\n        out\n    }"))

PORT_SANE = """            weights_sane(
                &out.weight.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                TOKENS,
                cfg.top_k,
                cfg.routed_scaling_factor as f64,
            ),"""

MUTANTS = [
    # ---- the primitive's own semantics -----------------------------------
    ("M01 combining weight taken from the BIASED score (the named trap)",
     "route: weight per expert", [(ROUTER,
      "        let prerenorm = self.combine_weights(&scores, &idx);",
      "        let prerenorm = self.combine_weights(&Scores::from_raw("
      "sfc.as_slice().to_vec(), tokens, self.cfg.num_experts), &idx);")]),

    ("M02 bias added to the LOGIT instead of the score",
     "route: selection + weight", [(ROUTER,
      "        let sfc = self.scores_for_choice(&scores);\n        let idx = self.select(&sfc);",
      "        let biased: Vec<f32> = logits.iter().enumerate()"
      ".map(|(i, &x)| x + self.bias[i % self.cfg.num_experts]).collect();\n"
      "        let sfc = ScoresForChoice::from_raw("
      "self.scores(&biased, tokens).as_slice().to_vec(), tokens, self.cfg.num_experts);\n"
      "        let idx = self.select(&sfc);")]),

    ("M03 correction bias dropped entirely",
     "scores_for_choice EXACT", [(ROUTER,
      "                v[t * n + e] = scores.v[t * n + e] + self.bias[e];",
      "                v[t * n + e] = scores.v[t * n + e];")]),

    ("M04 bias broadcast over the WRONG axis (per token, not per expert)",
     "scores_for_choice EXACT", [(ROUTER,
      "                v[t * n + e] = scores.v[t * n + e] + self.bias[e];",
      "                v[t * n + e] = scores.v[t * n + e] + self.bias[t % n];")]),

    ("M05 renormalize turned off in the shipping config",
     "config + weights", [(ROUTER, CFG_BLOCK,
      CFG_BLOCK.replace("renormalize: true,", "renormalize: false,"))]),

    ("M06 routed_scaling_factor 1.0 -> 2.0",
     "config + weights", [(ROUTER, CFG_BLOCK,
      CFG_BLOCK.replace("routed_scaling_factor: 1.0,", "routed_scaling_factor: 2.0,"))]),

    ("M07 top_k 16 -> 8",
     "config + shapes", [(ROUTER, CFG_BLOCK, CFG_BLOCK.replace("top_k: 16,", "top_k: 8,"))]),

    ("M08 num_experts 896 -> 895",
     "config + shapes", [(ROUTER, CFG_BLOCK,
      CFG_BLOCK.replace("num_experts: 896,", "num_experts: 895,"))]),

    ("M09 select takes the BOTTOM 16 (comparator reversed)",
     "SELECTION", [(ROUTER, "                sb.partial_cmp(&sa)", "                sa.partial_cmp(&sb)")]),

    ("M10 select off by one (ranks 2..17 instead of 1..16)",
     "SELECTION", [(ROUTER,
      "            idx[t * k..(t + 1) * k].copy_from_slice(&order[..k]);",
      "            idx[t * k..(t + 1) * k].copy_from_slice(&order[1..k + 1]);")]),

    ("M11 select emits the top expert 16 times (duplicates)",
     "idx well-formed", [(ROUTER,
      "            idx[t * k..(t + 1) * k].copy_from_slice(&order[..k]);",
      "            idx[t * k..(t + 1) * k].copy_from_slice(&vec![order[0]; k]);")]),

    ("M12 gather from expert e+1 (weights off by one expert)",
     "prerenorm EXACT", [(ROUTER,
      "                out[t * k + j] = scores.v[t * n + e];",
      "                out[t * k + j] = scores.v[t * n + (e + 1) % n];")]),

    ("M13 normalize divides by k instead of by the sum",
     "normalize", [(ROUTER, "                let d = s + 1e-20f32;", "                let d = k as f32;")]),

    ("M14 the +1e-20 epsilon inflated to +1e-2",
     "normalize", [(ROUTER, "                let d = s + 1e-20f32;", "                let d = s + 1e-2f32;")]),

    ("M15 logits read the NEXT expert's weight row",
     "logits", [(ROUTER,
      "                let w = &self.weight[e * hid..(e + 1) * hid];",
      "                let ee = (e + 1) % self.cfg.num_experts;\n"
      "                let w = &self.weight[ee * hid..(ee + 1) * hid];")]),

    ("M16 f64 accumulation lane drops the multiply",
     "logits f64", [(ROUTER,
      "                        .fold(0f64, |s, (&a, &b)| s + a as f64 * b as f64)",
      "                        .fold(0f64, |s, (&a, _b)| s + a as f64)")]),

    ("M17 f32 accumulation lane drops the multiply",
     "logits f32", [(ROUTER,
      "                    Accum::F32 => x.iter().zip(w).fold(0f32, |s, (&a, &b)| s + a * b),",
      "                    Accum::F32 => x.iter().zip(w).fold(0f32, |s, (&a, _b)| s + a),")]),

    ("M18 sigmoid replaced by the identity",
     "scores", [(ROUTER,
      "                .map(|&x| (1.0f64 / (1.0 + (-(x as f64)).exp())) as f32)",
      "                .map(|&x| x)")]),

    ("M19 bf16 widening shifts by 8 instead of 16",
     "logits (artifact hash cannot see this)", [(ROUTER,
      "    bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()",
      "    bits.iter().map(|&b| f32::from_bits((b as u32) << 8)).collect()")]),

    ("M20 grouped-routing refusal removed (silently routes as if ungrouped)",
     "port: grouped branch refused", [(ROUTER,
      "        if self.num_expert_group > 1 && self.num_expert_group > self.topk_group {",
      "        if false && self.num_expert_group > 1 && self.num_expert_group > self.topk_group {")]),

    # ---- the gate's own teeth --------------------------------------------
    ("M21 NaN injected into Router::normalize's output",
     "non-finite counter + port-weight sanity", [(ROUTER, NAN_EDIT[1], NAN_EDIT[2])]),

    ("M22 same NaN + the comparator relaxed to `d > maxabs`",
     "non-finite counter (the comparator alone never held it)",
     [NAN_EDIT, (GATE, "        if !(d <= maxabs) {", "        if d > maxabs {")]),

    ("M37 same NaN + the non-finite counter neutered",
     "port-weight sanity check alone",
     [NAN_EDIT, (GATE, "            nonfinite += 1;", "            nonfinite += 0;")]),

    ("M38 same NaN + counter neutered + port-weight sanity check neutered",
     "EXPECTED SURVIVOR — names exactly what stands between a NaN and a green gate",
     [NAN_EDIT,
      (GATE, "            nonfinite += 1;", "            nonfinite += 0;"),
      (GATE, PORT_SANE, "            true,")]),

    ("M23 LAYERS shortened to [1] (a lane silently dropped)",
     "totality + check-count floor", [(GATE,
      "const LAYERS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];",
      "const LAYERS: [usize; 1] = [1];")]),

    ("M24 the per-layer loop skipped entirely (a fold over nothing is vacuous)",
     "totality + check-count floor", [(GATE,
      "    for &l in LAYERS.iter() {\n        println!(\"--- layer {l} ---\");",
      "    for &l in LAYERS.iter().take(0) {\n        println!(\"--- layer {l} ---\");")]),

    ("M25 tie-margin measured as (17th - 16th), i.e. negative",
     "selection well-determined", [(GATE,
      "        let m = row[k - 1] - row[k];", "        let m = row[k] - row[k - 1];")]),

    ("M26 the bias-in-weight CONTROL rebuilt from the unbiased scores",
     "CONTROL reproduces ALT / misses truth", [(GATE,
      "    let wrong_pre = router.combine_weights(&sfc_as_scores, &ref_idx);",
      "    let wrong_pre = router.combine_weights(&Scores::from_raw("
      "ref_scores.iter().map(|&x| x as f32).collect(), TOKENS, cfg.num_experts), &ref_idx);")]),

    ("M27 the bias-on-logits CONTROL uses half the bias",
     "CONTROL reproduces ALT idx", [(GATE,
      "        .map(|(i, &x)| x as f32 + router.bias()[i % cfg.num_experts])",
      "        .map(|(i, &x)| x as f32 + 0.5 * router.bias()[i % cfg.num_experts])")]),

    ("M28 the 'bias changes the chosen experts' check fed the BIASED scores",
     "bias CHANGES chosen experts", [(GATE,
      "    let unbiased_sfc = ScoresForChoice::from_raw(\n"
      "        ref_scores.iter().map(|&x| x as f32).collect(),",
      "    let unbiased_sfc = ScoresForChoice::from_raw(\n"
      "        ref_sfc.iter().map(|&x| x as f32).collect(),")]),

    ("M29 set_equal_rows made vacuous AND select takes the bottom 16",
     "defense in depth: weight pairing must still catch it",
     [(GATE, "        if x == y {", "        if true {"),
      (ROUTER, "                sb.partial_cmp(&sa)", "                sa.partial_cmp(&sb)")]),
]

ARTIFACT_MUTANTS = [
    ("M30 layer 5 gate weights NEGATED in the exported npz",
     "artifact sha256 + logits", "neg", t_negate),
    ("M31 an exported array truncated to length 0",
     "EMPTY assert / premise shapes", "trunc", t_truncate),
    ("M32 the manifest's sha256 for one array falsified",
     "artifact sha256", "lie", t_manifest_lie),
    ("M33 the f32 bias replaced by its own bf16 rounding",
     "pinned f32-vs-bf16 divergence count", "f32eqbf16", t_bias_f32_is_bf16),
    ("M34 one bf16 bias value moved by 1 ulp",
     "artifact sha256 + oracle-bias equality", "perturb", t_bias_perturb),
    ("M35 an exported array missing entirely",
     "manifest count / npz lookup", "drop", t_drop_array),
]


def main():
    results = []
    print("=== UNMUTATED baseline ===", flush=True)
    restore()
    ok, err = build()
    assert ok, err
    rc, out, _ = run_gate()
    base_checks = [l for l in out.splitlines() if l.startswith("GATE ")]
    print(f"  exit={rc}  {base_checks}", flush=True)
    results.append(("UNMUTATED", "-", "PASS" if rc == 0 else "BROKEN", 0,
                    base_checks[0] if base_checks else ""))
    assert rc == 0, "baseline must pass before mutating"

    for name, target, edits in MUTANTS:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        restore()
        try:
            apply_edits(edits)
        except SystemExit as e:
            print(f"  {e}", flush=True)
            results.append((name, target, "ANCHOR-FAIL", 0, str(e)[:110]))
            continue
        ok, err = build()
        if not ok:
            print("  BUILD FAILED (not a valid mutant):\n" + err[-800:], flush=True)
            results.append((name, target, "BUILD-FAIL", 0, ""))
            continue
        rc, out, serr = run_gate()
        f = failures(out)
        panicked = "panicked" in serr
        verdict = "CAUGHT" if rc != 0 else "SURVIVED"
        detail = (f[0].strip()[:110] if f else
                  (serr.strip().splitlines()[-1][:110] if panicked else ""))
        print(f"  exit={rc} failures={len(f)} panic={panicked} -> {verdict} "
              f"({time.time()-t0:.0f}s)", flush=True)
        for l in f[:4]:
            print("    " + l.strip()[:150], flush=True)
        if panicked:
            print("    PANIC: " + serr.strip().splitlines()[-1][:150], flush=True)
        results.append((name, target, verdict, len(f) + (1 if panicked else 0), detail))

    restore()
    ok, err = build()
    assert ok, err

    for name, target, tag, transform in ARTIFACT_MUTANTS:
        print(f"\n=== {name} ===", flush=True)
        d = make_mut_vecdir(tag, transform)
        rc, out, serr = run_gate(d)
        f = failures(out)
        panicked = "panicked" in serr
        verdict = "CAUGHT" if rc != 0 else "SURVIVED"
        detail = (f[0].strip()[:110] if f else
                  (serr.strip().splitlines()[-1][:110] if panicked else ""))
        print(f"  exit={rc} failures={len(f)} panic={panicked} -> {verdict}", flush=True)
        for l in f[:4]:
            print("    " + l.strip()[:150], flush=True)
        if panicked:
            print("    PANIC: " + serr.strip().splitlines()[-1][:150], flush=True)
        results.append((name, target, verdict, len(f) + (1 if panicked else 0), detail))
        shutil.rmtree(d, ignore_errors=True)

    # ---- absent file ------------------------------------------------------
    print("\n=== M36 the vectors directory does not exist ===", flush=True)
    rc, out, serr = run_gate(os.path.join(WORK, "does_not_exist"))
    verdict = "CAUGHT" if rc != 0 else "SURVIVED"
    print(f"  exit={rc} -> {verdict}: {(out+serr).strip().splitlines()[-1][:120]}", flush=True)
    results.append(("M36 the vectors directory does not exist", "absent file",
                    verdict, 1, (out + serr).strip().splitlines()[-1][:110]))

    restore()
    ok, err = build()
    assert ok, err
    rc, out, _ = run_gate()
    print(f"\n=== reverted: exit={rc} "
          f"{[l for l in out.splitlines() if l.startswith('GATE ')]} ===", flush=True)

    json.dump(results, open(os.path.join(WORK, "mutants_router.json"), "w"), indent=1)
    print("\n" + "=" * 110)
    print(f"{'mutant':<66} {'targets':<44} verdict")
    for n, t, v, nf, d in results:
        print(f"{n:<66} {t:<44} {v} ({nf})")
    surv = [r for r in results if r[2] == "SURVIVED"]
    print(f"\n{len(results)-1} mutants, {len([r for r in results if r[2]=='CAUGHT'])} caught, "
          f"{len(surv)} survived")
    for r in surv:
        print("  SURVIVOR: " + r[0])


if __name__ == "__main__":
    try:
        main()
    finally:
        restore()
