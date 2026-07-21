#!/usr/bin/env python3
"""Compare F5 probe tensors: reference (probes/python) vs Burn (probes/rust)."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "probes", "python")
RS = os.path.join(ROOT, "probes", "rust")

# in forward order, so the first DIVERGED row localises the bug
PROBES = ["text_embed", "time_embed", "input_embed", "block0", "block21", "norm_out", "output"]


def stats(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    diff = np.abs(a - b)
    rn = np.linalg.norm(a)
    rel = np.linalg.norm(diff) / rn if rn > 0 else float("inf")
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    return diff.max(), rel, cos


print(f"{'probe':<12} {'shape':<16} {'max_abs':>10} {'rel_err':>10} {'cosine':>9}  status")
print("-" * 70)
for name in PROBES:
    pp, rp = os.path.join(PY, name + ".npy"), os.path.join(RS, name + ".npy")
    if not (os.path.exists(pp) and os.path.exists(rp)):
        print(f"{name:<12} (missing)")
        continue
    pa, ra = np.load(pp), np.load(rp)
    if pa.shape != ra.shape:
        print(f"{name:<12} SHAPE MISMATCH py{pa.shape} rs{ra.shape}")
        continue
    mx, rel, cos = stats(pa, ra)
    status = "EXCELLENT" if rel < 1e-3 else "GOOD" if rel < 1e-2 else "FAIR" if rel < 1e-1 else "DIVERGED"
    print(f"{name:<12} {str(pa.shape):<16} {mx:>10.2e} {rel:>10.2e} {cos:>9.5f}  {status}")
