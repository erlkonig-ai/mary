#!/usr/bin/env python3
"""Cut named tensors -- optionally gathering arbitrary leading-axis indices --
out of a safetensors checkpoint into a small safetensors directory.

Extends `inkling_slice_extract.py`, which can only take a leading PREFIX. A MoE
layer's firing experts are scattered across the expert axis (34 of 256, up to
index 249), so a prefix would have to copy the whole 3.2 GB stack to reach them.
Gathering the 34 costs 480 MB instead.

Stdlib only, on purpose: the machine holding the checkpoint need not have torch.
Reads the safetensors header directly -- 8-byte little-endian header length,
then a JSON directory of {dtype, shape, data_offsets}.

  usage: inkling_gather_extract.py <ckpt dir> <out dir> <spec> ...
    spec = NAME                 whole tensor
         | NAME@i,j,k           gather those leading-axis indices
"""
import json
import os
import struct
import sys

BYTES_PER = {
    "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
    "F64": 8, "I64": 8, "U64": 8,
}


def header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def read_tensor(path, meta, base, idx):
    """Whole tensor if `idx` is None, else those leading-axis rows in order."""
    shape = list(meta["shape"])
    esz = BYTES_PER[meta["dtype"]]
    start = base + meta["data_offsets"][0]
    if idx is None:
        n = meta["data_offsets"][1] - meta["data_offsets"][0]
        with open(path, "rb") as f:
            f.seek(start)
            return meta["dtype"], shape, f.read(n)
    # Bytes per leading-axis row.
    row = esz
    for s in shape[1:]:
        row *= s
    out = bytearray()
    with open(path, "rb") as f:
        for i in idx:
            if not 0 <= i < shape[0]:
                raise SystemExit("index %d out of range for %s" % (i, shape))
            f.seek(start + i * row)
            out.extend(f.read(row))
    return meta["dtype"], [len(idx)] + shape[1:], bytes(out)


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    specs = sys.argv[3:]
    if not specs:
        sys.exit(__doc__)
    os.makedirs(out, exist_ok=True)
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]

    tensors, blob, gathered = {}, bytearray(), {}
    for spec in specs:
        name, _, sel = spec.partition("@")
        idx = [int(p) for p in sel.split(",")] if sel else None
        shard = index[name]
        meta, base = header(os.path.join(ckpt, shard))
        dtype, shape, data = read_tensor(os.path.join(ckpt, shard), meta[name], base, idx)
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob.extend(data)
        if idx is not None:
            gathered[name] = idx
        print("%-64s %-8s %s -> %s (%.1f MB)"
              % (name, dtype, meta[name]["shape"], shape, len(data) / 1e6))

    hdr = json.dumps(tensors, separators=(",", ":")).encode()
    hdr += b" " * ((-len(hdr)) % 8)
    shard = "model-00001-of-00001.safetensors"
    with open(os.path.join(out, shard), "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(blob)
    json.dump({"weight_map": {k: shard for k in tensors}},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    # Which original rows each gathered tensor kept, so the consumer can scatter
    # them back to their true expert ids instead of guessing.
    json.dump(gathered, open(os.path.join(out, "gathered_indices.json"), "w"), indent=1)
    print("wrote %s (%.1f MB)" % (out, (8 + len(hdr) + len(blob)) / 1e6))


main()
