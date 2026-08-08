#!/usr/bin/env python3
"""Causality gate: a token must not influence any position before it.

Every other gate in this port compares mary's masks against the reference's
masks, which would agree even if BOTH leaked future information. This checks the
property directly and end to end: run the forward on a prefix, run it again on
that prefix plus one more token, and require the shared prefix's hidden states
to be BITWISE identical at every sampled layer.

Bitwise really is the bar. The two runs do identical arithmetic on identical
inputs for those positions — if attention is causal. Any difference at all,
however small, means information moved backwards.

It also underwrites the caching work: if the prefix is invariant then prefix
routing is invariant, so a decoded-expert cache and a KV cache are both sound.

  usage: inkling_causal_gate.py <forward binary> <ckpt> <ids.bin> [extra_token]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

BIN, CKPT, IDS = sys.argv[1], sys.argv[2], sys.argv[3]
EXTRA = int(sys.argv[4]) if len(sys.argv) > 4 else 12650
LAYERS = [0, 1, 2, 5, 20, 41]

ids = list(np.frombuffer(open(IDS, "rb").read(), dtype="<i8"))
print("=== causality gate ===")
print("  prefix     : %d tokens %s" % (len(ids), ids))
print("  plus token : %d" % EXTRA)

tmp = tempfile.mkdtemp(prefix="inkcausal_")
short_ids = os.path.join(tmp, "short.bin")
long_ids = os.path.join(tmp, "long.bin")
open(short_ids, "wb").write(np.array(ids, dtype="<i8").tobytes())
open(long_ids, "wb").write(np.array(ids + [EXTRA], dtype="<i8").tobytes())

runs = {}
for tag, idfile in (("short", short_ids), ("long", long_ids)):
    d = os.path.join(tmp, tag)
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, INK_DUMP_DIR=d)
    r = subprocess.run([BIN, CKPT, idfile, os.path.join(tmp, tag + "_top.bin")],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        sys.exit("forward failed for %s" % tag)
    runs[tag] = d

H = None
compared = 0
bad = []
for L in LAYERS:
    a_p = os.path.join(runs["short"], "h_after_%02d.bin" % L)
    b_p = os.path.join(runs["long"], "h_after_%02d.bin" % L)
    if not (os.path.exists(a_p) and os.path.exists(b_p)):
        print("  layer %2d: no dump — the forward may not have that many layers" % L)
        continue
    a = np.frombuffer(open(a_p, "rb").read(), dtype="<f4")
    b = np.frombuffer(open(b_p, "rb").read(), dtype="<f4")
    if H is None:
        H = a.size // len(ids)
    a = a.reshape(-1, H)
    b = b.reshape(-1, H)
    k = a.shape[0]
    same = np.array_equal(a, b[:k])
    compared += a.size
    if not same:
        bad.append((L, float(np.abs(a - b[:k]).max())))
    print("  layer %2d: prefix %s vs first %d of %s -> %s"
          % (L, a.shape, k, b.shape,
             "identical" if same else "DIFFERS max %.3e" % np.abs(a - b[:k]).max()))

print("\n  layers compared : %d" % (len(LAYERS) - 0))
print("  values compared : %d" % compared)

if compared == 0:
    sys.exit("GATE VACUOUS — compared nothing; the forward wrote no dumps")
if bad:
    for L, d in bad:
        print("  FAIL  layer %d differs by %.3e — information moved backwards" % (L, d))
    sys.exit("GATE FAILED — %d of %d layers leak" % (len(bad), len(LAYERS)))
print("GATE PASSED — %d values bitwise identical; attention is causal end to end"
      % compared)
