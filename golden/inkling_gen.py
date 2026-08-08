#!/usr/bin/env python3
"""Greedy multi-token generation by re-running the forward.

No KV cache and no change to the binary: inkling_forward already emits the
top-5 for every position, so the last position's top-1 is the next token.
Recomputing the whole prefix each step is O(n^2) and slow, and it is also
obviously correct, which matters more here — the point is a continuation that
reads as a sentence, which is much stronger evidence than a single argmax.

  ink_gen.py <ckpt> <bin dir> <prompt> <n_new>
"""
import os
import subprocess
import sys

import numpy as np
from transformers import AutoTokenizer

CKPT, BIN, PROMPT, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
ids = tok(PROMPT, return_tensors=None)["input_ids"]
print("prompt: %r" % PROMPT)
print("ids   : %s" % ids)

for step in range(N):
    open("/tmp/gen_ids.bin", "wb").write(np.array(ids, dtype="<i8").tobytes())
    r = subprocess.run(
        [os.path.join(BIN, "inkling_forward"), CKPT, "/tmp/gen_ids.bin", "/tmp/gen_top5.bin"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("forward failed at step %d" % step)
    top = np.frombuffer(open("/tmp/gen_top5.bin", "rb").read(), dtype="<i8")
    nxt = int(top.reshape(-1, 5)[-1, 0])          # last position, rank 0
    ids.append(nxt)
    print("  step %2d: +%-8d %-16r  ->  %s"
          % (step, nxt, tok.decode([nxt]), repr(tok.decode(ids))))

print("\nFINAL: %r" % tok.decode(ids))
