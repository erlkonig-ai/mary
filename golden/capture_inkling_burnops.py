#!/usr/bin/env python3
"""Python oracle for the Burn lane's ops, at REAL model dimensions.

The Burn gate was comparing expert_ffn and dense_mlp against the slice lane
because no Python dump existed at these shapes. That is a reference of a
reference, and the slice lane is the slow one. This dumps torch's own answer so
Burn can be checked against it directly.

Real dimensions on purpose: hidden 4096, per-expert intermediate 2048, dense
intermediate 16384. A toy size would not exercise a backend matmul's blocking,
which is the only reason the lanes differ at all -- the same reasoning that put
the existing gate at real widths.

Weights are dumped alongside the outputs so the gate reads exactly the numbers
torch saw, rather than regenerating "the same" random values on two sides and
hoping the generators agree.

  usage: capture_inkling_burnops.py <out dir>
"""
import os
import sys

import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "./inkling-oracle"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(20260808)
torch.set_default_dtype(torch.float32)

T, H, INTER, DENSE = 8, 4096, 2048, 16384


def w(name, t):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
    open(os.path.join(OUT, name), "wb").write(a.tobytes())
    return a.size


x = torch.randn(T, H) * 0.1
total = w("bop_x.bin", x)

# ---- one expert's feed-forward: down(silu(gate) * up) --------------------
# gate_up is [2 * inter, hidden] with the GATE ROWS FIRST, which is what
# load::deinterleave_fused produces from the interleaved checkpoint layout.
gate_up = torch.randn(2 * INTER, H) * 0.02
down = torch.randn(H, INTER) * 0.02
with torch.no_grad():
    both = torch.nn.functional.linear(x, gate_up)
    g, u = both[:, :INTER], both[:, INTER:]
    y_expert = torch.nn.functional.linear(torch.nn.functional.silu(g) * u, down)
total += w("bop_expert_gate_up.bin", gate_up)
total += w("bop_expert_down.bin", down)
total += w("bop_expert_y.bin", y_expert)
print("expert_ffn : x %s -> %s  |y| max %.6g" % (tuple(x.shape), tuple(y_expert.shape), y_expert.abs().max()))

# ---- dense MLP: down(silu(gate(x)) * up(x)) * global_scale ---------------
d_gate = torch.randn(DENSE, H) * 0.02
d_up = torch.randn(DENSE, H) * 0.02
d_down = torch.randn(H, DENSE) * 0.02
GS = 1.7
with torch.no_grad():
    gg = torch.nn.functional.linear(x, d_gate)
    uu = torch.nn.functional.linear(x, d_up)
    y_dense = torch.nn.functional.linear(torch.nn.functional.silu(gg) * uu, d_down) * GS
total += w("bop_dense_gate.bin", d_gate)
total += w("bop_dense_up.bin", d_up)
total += w("bop_dense_down.bin", d_down)
total += w("bop_dense_y.bin", y_dense)
print("dense_mlp  : x %s -> %s  |y| max %.6g" % (tuple(x.shape), tuple(y_dense.shape), y_dense.abs().max()))

manifest = {
    "tokens": T, "hidden": H, "intermediate": INTER, "dense_intermediate": DENSE,
    "global_scale": GS,
}
import json
json.dump(manifest, open(os.path.join(OUT, "bop_manifest.json"), "w"), indent=1)
print("wrote %d floats (%.2f GB) to %s" % (total, total * 4 / 1e9, OUT))
