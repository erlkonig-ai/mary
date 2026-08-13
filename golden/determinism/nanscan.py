"""Report the FIRST non-finite element in a dump directory, and how many.

The first NaN, not the first token flip. A flipped argmax is a lagging
indicator -- it tells you a value moved, several layers after it moved and
after a softmax has already thrown away the magnitude. `h_after_NN.bin` is the
residual stream itself, so the layer named here is the layer that produced the
NaN and not the layer that noticed.
"""

import glob
import os
import sys

import numpy as np

d = sys.argv[1]
first = None
total = 0

paths = sorted(glob.glob(os.path.join(d, "h_after_*.bin")))
paths.append(os.path.join(d, "h_embed.bin"))
for p in paths:
    if not os.path.exists(p):
        continue
    a = np.fromfile(p, dtype=np.float32)
    bad = ~np.isfinite(a)
    n = int(bad.sum())
    if n:
        total += n
        if first is None:
            idx = int(np.flatnonzero(bad)[0])
            first = f"{os.path.basename(p)}@row{idx // 4096}col{idx % 4096}"

print(f"nonfinite={total}" + (f" first={first}" if first else ""))
