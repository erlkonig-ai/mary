#!/usr/bin/env python3
"""Cut a few named tensor slices out of a checkpoint into a tiny safetensors dir.

Why this exists: the NVFP4 oracle
(`capture_inkling_nvfp4.py`) needs `compressed_tensors`, which needs torch, and
the machine that holds the 160 GB checkpoint is not always the machine that has
torch. This reads the safetensors header with the standard library only —
8-byte little-endian header length, then a JSON directory of
`{dtype, shape, data_offsets}` — copies the requested leading slices, and writes
a directory that `capture_inkling_nvfp4.py` accepts unmodified: one shard plus a
`model.safetensors.index.json` naming it.

Slicing is leading-axis only, which is all a row-major layout makes contiguous
and all the oracle asks for.

  usage: inkling_slice_extract.py <ckpt dir> <out dir> <name>[:e[:r]] ...
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


def read_slice(path, meta, base, dims):
    """Leading-axis slice: `dims` gives the kept length of each leading axis."""
    shape = meta["shape"]
    dtype = meta["dtype"]
    esz = BYTES_PER[dtype]
    keep = list(shape)
    for i, d in enumerate(dims):
        keep[i] = min(d, shape[i])
    # Stride, in elements, of each axis.
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.insert(0, acc)
        acc *= s
    start = base + meta["data_offsets"][0]
    out = bytearray()
    with open(path, "rb") as f:
        # Walk the kept prefix of every axis except the last, which is whole.
        def walk(axis, off):
            if axis == len(shape) - 1:
                f.seek(start + off * esz)
                out.extend(f.read(keep[axis] * esz))
                return
            for i in range(keep[axis]):
                walk(axis + 1, off + i * strides[axis])

        walk(0, 0)
    return dtype, keep, bytes(out)


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    specs = sys.argv[3:]
    if not specs:
        sys.exit(__doc__)
    os.makedirs(out, exist_ok=True)
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]

    tensors, blob = {}, bytearray()
    for spec in specs:
        parts = spec.split(":")
        name = parts[0]
        dims = [int(p) for p in parts[1:]]
        shard = index[name]
        meta, base = header(os.path.join(ckpt, shard))
        dtype, shape, data = read_slice(os.path.join(ckpt, shard), meta[name], base, dims)
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob.extend(data)
        print("%-64s %-8s %s -> %s (%d bytes)" % (name, dtype, meta[name]["shape"], shape, len(data)))

    hdr = json.dumps(tensors, separators=(",", ":")).encode()
    pad = (-len(hdr)) % 8
    hdr += b" " * pad
    shard = "model-00001-of-00001.safetensors"
    with open(os.path.join(out, shard), "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(blob)
    json.dump({"weight_map": {k: shard for k in tensors}},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    print("wrote %s (%.1f KB)" % (out, (8 + len(hdr) + len(blob)) / 1e3))


main()
