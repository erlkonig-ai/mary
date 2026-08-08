#!/usr/bin/env python3
"""Emit NVFP4 reference vectors as raw bins for the Rust gate.

The block scales go out as their RAW E4M3 bytes, not as pre-converted floats:
handing Rust a float32 scale would test the multiply and skip the FP8 decode,
which is half the work and the half more likely to be wrong. The Rust side has
to turn 0x?? into a number by itself and agree with torch about the result.

Nibble order was settled separately by compressed_tensors (low-first) and is
recorded in the manifest so the gate fails loudly if that ever changes.
"""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open
from compressed_tensors.compressors import unpack_fp4_from_uint8

D = sys.argv[1] if len(sys.argv) > 1 else "./Inkling-Small-NVFP4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "./inkling-oracle"

import os
os.makedirs(OUT, exist_ok=True)

weight_map = json.load(open(D + "/model.safetensors.index.json"))["weight_map"]


def get(name, sl=None):
    with safe_open(D + "/" + weight_map[name], framework="pt") as f:
        return f.get_tensor(name) if sl is None else f.get_slice(name)[sl]


BASE = "model.llm.layers.10.mlp.experts.w13_weight"
E, R = 3, 4

codes = get(BASE, (slice(0, E), slice(0, R), slice(None))).contiguous()
scale = get(BASE + ".scale", (slice(0, E), slice(0, R), slice(None))).contiguous()
scale2 = get(BASE + ".scale2")[:E].float().contiguous()

n_bytes = codes.shape[-1]
n_logical = n_bytes * 2
n_scales = scale.shape[-1]
group = n_logical // n_scales
assert group == 16, group

# The authority does the unpacking; the scales and scale2 are applied here.
vals = unpack_fp4_from_uint8(codes.reshape(-1, n_bytes), E * R, n_logical,
                             dtype=torch.float32).reshape(E, R, n_logical)
deq = vals.reshape(E, R, n_scales, group) * scale.float()[..., None]
deq = deq.reshape(E, R, n_logical) * scale2[:, None, None]
deq = deq.float().numpy().astype(np.float32)

# Raw bytes, exactly as stored.
codes_u8 = codes.numpy().astype(np.uint8)
scale_u8 = scale.view(torch.uint8).numpy().astype(np.uint8)

open(OUT + "/nvfp4_codes.bin", "wb").write(codes_u8.tobytes())
open(OUT + "/nvfp4_scale_e4m3.bin", "wb").write(scale_u8.tobytes())
open(OUT + "/nvfp4_scale2_f32.bin", "wb").write(scale2.numpy().astype("<f4").tobytes())
open(OUT + "/nvfp4_expected_f32.bin", "wb").write(deq.astype("<f4").tobytes())

# Every distinct E4M3 byte that actually occurs, so the gate can report how much
# of the FP8 domain its agreement actually covers.
present = sorted(int(x) for x in np.unique(scale_u8))
manifest = {
    "tensor": BASE,
    "experts": E, "rows": R,
    "bytes_per_row": n_bytes,
    "logical_per_row": n_logical,
    "scales_per_row": n_scales,
    "group": group,
    "nibble_order": "low-first",
    "authority": "compressed_tensors.compressors.unpack_fp4_from_uint8",
    "distinct_e4m3_bytes_present": present,
    "expected_abs_max": float(np.abs(deq).max()),
    "expected_std": float(deq.std()),
}
json.dump(manifest, open(OUT + "/nvfp4_manifest.json", "w"), indent=1)

print("wrote to", OUT)
for k, v in manifest.items():
    if k == "distinct_e4m3_bytes_present":
        print("  %-28s %d distinct values" % (k, len(v)))
    else:
        print("  %-28s %s" % (k, v))
print("  values                       %d" % deq.size)

# A full-domain E4M3 table too: all 256 byte patterns and torch's float for each.
# Without it the gate only ever checks the handful of scales this slice happens
# to use, and would call that "the FP8 decode agrees".
allb = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
open(OUT + "/e4m3_table_f32.bin", "wb").write(allb.astype("<f4").tobytes())
print("  e4m3 full table              256 entries, finite: %d" % int(np.isfinite(allb).sum()))
