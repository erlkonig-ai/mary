# Unpacks the Kimi K3 `situ` oracle into the form `kimi_situ_gate` reads.
#
# The oracle itself is NOT produced here: `situ_activation.npz` was captured by
# instantiating Moonshot's shipped `SituAndMul` module (modeling_kimi_linear.py)
# and running its forward pass, plus an independent float64 transcription of
# the same formula, over a 1868-point sweep, a 34x34 sign-quadrant grid and one
# MoE-shaped 8x6144 block. This script only splits that .npz (a zip of .npy
# members) into individual .npy files, so the Rust gate needs no zip reader --
# and asserts the source hash first, because a silently-replaced oracle is the
# one failure mode a gate cannot catch by itself.
#
#   python golden/kimi_situ_unpack.py <situ_activation.npz> <out_dir>
#
# Defaults match the Spark layout the gate defaults to.
import hashlib
import json
import os
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("K3_ORACLE_DIR", "./k3-oracle"), "situ_activation.npz")
DST = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("K3_SITU_OUT", "./k3-situ/oracle_npy")

# sha256 of the capture this port was gated against (2026-08-05).
EXPECT = "e22d1ba33e19367c5b5484d791fdf1ff0e239684c084704a84544b0b5f001ed7"

digest = hashlib.sha256(open(SRC, "rb").read()).hexdigest()
if digest != EXPECT:
    raise SystemExit(
        f"{SRC}\n  sha256 {digest}\n  expected {EXPECT}\n"
        "The oracle is not the capture this gate was written against. Re-run the\n"
        "gate knowingly, or update EXPECT together with the gate's numbers -- do\n"
        "not silently accept a different oracle."
    )

os.makedirs(DST, exist_ok=True)
z = np.load(SRC)
index = {"source": SRC, "source_sha256": digest, "arrays": {}}
for k in z.files:
    a = np.ascontiguousarray(z[k])
    p = os.path.join(DST, k + ".npy")
    np.save(p, a)
    # round-trip, so a transcode bug cannot reach the gate as "oracle data"
    b = np.load(p)
    assert b.dtype == a.dtype and b.shape == a.shape and a.tobytes() == b.tobytes(), k
    index["arrays"][k] = {"shape": list(a.shape), "dtype": str(a.dtype)}
json.dump(index, open(os.path.join(DST, "index.json"), "w"), indent=1)
print(f"unpacked {len(z.files)} arrays -> {DST}")
