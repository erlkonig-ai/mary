#!/usr/bin/env python3
"""Compare Vocos probes: reference (probes/vocos) vs Burn (probes/vocos_rust)."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "probes", "vocos")
RS = os.path.join(ROOT, "probes", "vocos_rust")
PROBES = ["embed", "backbone", "head_out", "audio"]

print(f"{'probe':<10} {'shape':<16} {'max_abs':>10} {'rel_err':>10} {'cosine':>9}  status")
print("-" * 68)
for name in PROBES:
    pp, rp = os.path.join(PY, name + ".npy"), os.path.join(RS, name + ".npy")
    if not (os.path.exists(pp) and os.path.exists(rp)):
        print(f"{name:<10} (missing)")
        continue
    a, b = np.load(pp).flatten().astype(np.float64), np.load(rp).flatten().astype(np.float64)
    if a.shape != b.shape:
        print(f"{name:<10} SHAPE py{np.load(pp).shape} rs{np.load(rp).shape}")
        continue
    diff = np.abs(a - b)
    rel = np.linalg.norm(diff) / (np.linalg.norm(a) + 1e-30)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    status = "EXCELLENT" if rel < 1e-3 else "GOOD" if rel < 1e-2 else "FAIR" if rel < 1e-1 else "DIVERGED"
    print(f"{name:<10} {str(np.load(pp).shape):<16} {diff.max():>10.2e} {rel:>10.2e} {cos:>9.5f}  {status}")
