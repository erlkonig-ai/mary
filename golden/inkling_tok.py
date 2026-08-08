#!/usr/bin/env python3
"""Tokenize prompts for the forward, and decode what it predicts.

The tokenizer is not under test — the question is whether mary's forward turns
real weights into a sensible continuation.

NOTE the filename: calling this `tokenize.py` shadows the stdlib `tokenize`
module, which `inspect` imports, and breaks the whole import graph with a
confusing "partially initialized module" error.

  ink_tok.py encode <ckpt> <out.bin> <prompt>     raw completion
  ink_tok.py chat   <ckpt> <out.bin> <prompt>     the checkpoint's own template
  ink_tok.py decode <ckpt> <ids.bin> <n_per_pos>
"""
import sys

import numpy as np
from transformers import AutoTokenizer

mode, ckpt = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

if mode == "encode":
    out, prompt = sys.argv[3], sys.argv[4]
    ids = tok(prompt, return_tensors=None)["input_ids"]
    open(out, "wb").write(np.array(ids, dtype="<i8").tobytes())
    print("prompt : %r" % prompt)
    print("ids    : %s" % ids)
    print("pieces : %s" % [tok.decode([i]) for i in ids])

elif mode == "chat":
    out, prompt = sys.argv[3], sys.argv[4]
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    open(out, "wb").write(np.array(ids, dtype="<i8").tobytes())
    print("templated: %r" % text[:400])
    print("ids (%d): %s" % (len(ids), ids[:48]))
    print("bos=%r eos=%r" % (tok.bos_token, tok.eos_token))

elif mode == "decode":
    ids = np.frombuffer(open(sys.argv[3], "rb").read(), dtype="<i8")
    k = int(sys.argv[4])
    for pos in range(len(ids) // k):
        row = ids[pos * k:(pos + 1) * k]
        print("  pos %d top%d: %s" % (pos, k, [repr(tok.decode([int(i)])) for i in row]))

else:
    sys.exit("unknown mode %r" % mode)
