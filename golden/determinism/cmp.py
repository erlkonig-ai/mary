"""Compare residual-stream dumps BITWISE, and say what SHAPE the difference is.

A max-absolute-error is the wrong instrument for this question. Two runs that
differ because floating-point addition is not associative differ a LITTLE in
ALMOST EVERY element; two runs that differ because one of them multiplied by a
weight that was not there differ a LOT in a HANDFUL of elements, and the
handful lies in whichever rows routed to the affected expert. Those two have
similar maxima and completely different row/column structure, so the structure
is what this prints.

    cmp.py <ref.bin> <other.bin> [<other.bin> ...]
"""

import sys

import numpy as np

paths = sys.argv[1:]
arrs = [np.fromfile(p, dtype=np.float32) for p in paths]
base = arrs[0]
n = base.size
print(f"reference: {paths[0]}  ({n} elements)")

for path, a in zip(paths[1:], arrs[1:]):
    bb = base.view(np.uint32)
    ab = a.view(np.uint32)
    d = bb != ab
    ndiff = int(d.sum())
    if ndiff == 0:
        print(f"  {path}: BITWISE IDENTICAL")
        continue
    ulp = np.abs(bb.astype(np.int64) - ab.astype(np.int64))[d]
    absd = np.abs(base[d] - a[d])
    rel = absd / np.maximum(np.abs(base[d]), 1e-30)
    print(f"  {path}: {ndiff}/{n} words differ ({100 * ndiff / n:.3f}%)")
    print(f"     ULP gap   min {ulp.min()} med {int(np.median(ulp))} max {ulp.max()}")
    print(f"     |abs|     max {absd.max():.6g}   |rel| max {rel.max():.6g} med {np.median(rel):.3g}")
    if n % 4096 == 0:
        rows = n // 4096
        m = d.reshape(rows, 4096)
        per_row = m.sum(1)
        nz = np.nonzero(per_row)[0]
        print(
            f"     rows touched {len(nz)}/{rows}  first {nz[:12].tolist()}  "
            f"cols per touched row: min {per_row[nz].min()} max {per_row[nz].max()}"
        )
        print(
            "     -> a sparse, fixed number of columns over a subset of rows is a WEIGHT "
            "that was not there;\n"
            "        broadband perturbation of nearly every element would be reduction order."
        )
