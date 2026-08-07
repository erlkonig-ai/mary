"""Export the Kimi-K3 MoE router weight + correction bias for layers 1..12 out
of the checkpoint shards, into ONE uncompressed .npz that the Rust gate reads.

The checkpoint is opened READ-ONLY; nothing under $MODEL is written.

Artifact discipline (gate requirement 3): the sidecar manifest records, for
every array, a SHA-256 over the array's raw little-endian bytes AS READ FROM
THE SAFETENSORS SHARD.  The Rust gate recomputes SHA-256 over the bytes it
parsed out of the .npz.  Agreement pins the artifact to its source across two
independent implementations of the digest (Python hashlib vs the `sha2` crate)
and two independent parsers (safetensors vs mary's npz reader).  It is NOT a
round trip of my own writer against my own reader.
"""

import hashlib
import json
import os

import numpy as np
import torch
from safetensors import safe_open

MODEL = os.environ.get("K3_MODEL_DIR", "./kimi-k3")
OUT = os.path.join(os.environ.get("K3_ORACLE_DIR", "./k3-oracle"),
              "k3router_gateweights_routerport.npz")
SIDE = os.path.join(os.environ.get("K3_ORACLE_DIR", "./k3-oracle"),
               "k3router_gateweights_routerport_manifest.json")
LAYERS = list(range(1, 13))

index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]

arrays = {}
manifest = {
    "source_model_dir": MODEL,
    "layers": LAYERS,
    "digest": "sha256 over the array's raw little-endian bytes, computed from "
              "the safetensors shard (not from the .npz)",
    "arrays": {},
}

def record(key, arr, name, shard, note):
    arrays[key] = arr
    raw = arr.tobytes()
    manifest["arrays"][key] = {
        "tensor": name,
        "shard": shard,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "note": note,
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    print(key, arr.dtype, arr.shape, manifest["arrays"][key]["sha256"][:16], flush=True)


dtypes_seen = {}
for L in LAYERS:
    # --- gate.weight: bf16 on disk, kept as its raw bit pattern -------------
    name = f"language_model.model.layers.{L}.block_sparse_moe.gate.weight"
    shard = index[name]
    with safe_open(os.path.join(MODEL, shard), framework="pt") as f:
        t = f.get_tensor(name)
    dtypes_seen[f"L{L:02d}.weight"] = str(t.dtype)
    assert t.dtype == torch.bfloat16, (name, t.dtype)
    u16 = t.view(torch.int16).cpu().numpy().astype(np.uint16)
    record(f"L{L:02d}_gate_weight_bf16bits", u16, name, shard,
           "bfloat16 bit pattern, exactly as stored on disk")

    # --- gate.e_score_correction_bias --------------------------------------
    # MEASURED SURPRISE: this one is FLOAT32 on disk, not bfloat16.  A
    # `dtype=bfloat16` model load rounds it down, and the whole-layer oracle was
    # captured that way, so BOTH are exported: the on-disk f32 (what the
    # checkpoint says) and its bf16 rounding (what the oracle's model held).
    name = f"language_model.model.layers.{L}.block_sparse_moe.gate.e_score_correction_bias"
    shard = index[name]
    with safe_open(os.path.join(MODEL, shard), framework="pt") as f:
        t = f.get_tensor(name)
    dtypes_seen[f"L{L:02d}.bias"] = str(t.dtype)
    assert t.dtype == torch.float32, (name, t.dtype)
    f32 = t.cpu().numpy().astype(np.float32)
    record(f"L{L:02d}_gate_bias_f32", f32, name, shard,
           "float32, exactly as stored on disk (CHECKPOINT TRUTH)")
    b16 = t.to(torch.bfloat16).view(torch.int16).cpu().numpy().astype(np.uint16)
    record(f"L{L:02d}_gate_bias_bf16bits", b16, name, shard,
           "the same tensor rounded to bfloat16 — what a dtype=bfloat16 load "
           "holds, and what the whole-layer oracle used")

manifest["checkpoint_dtypes"] = dtypes_seen
# every layer, all three arrays, nothing missing
assert len(arrays) == 3 * len(LAYERS), len(arrays)

np.savez(OUT, **arrays)
json.dump(manifest, open(SIDE, "w"), indent=1)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
print("wrote", SIDE)
