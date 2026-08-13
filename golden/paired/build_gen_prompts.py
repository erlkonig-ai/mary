#!/usr/bin/env python3
"""Open-ended prompts for the teacher-forced half, tokenised by `inkling_encode`.

The multiple-choice half spends one whole NVFP4 process — about four minutes,
almost all of it the two halves hashing their expert slabs before they compute
anything — to obtain ONE paired token. The teacher-forced half spends the same
four minutes and obtains as many paired tokens as it generates. At `INK_GEN=48`
that is a forty-eight-fold better rate of paired observations per unit of
compute, which makes it the statistically efficient measurement and the
multiple-choice set the interpretable one. Both, therefore.

Ids come from `inkling_encode` — the Rust `tokenizers` executor over the
checkpoint's own `tokenizer.json` — rather than from a python one-liner, and
each ids file keeps the manifest `inkling_encode` writes beside it. That tool
exists because a gate in this tree asserted one prompt while its ids file held
another and nothing ever decoded the file to notice; an ids file that carries
its own text, and a decode round trip checked at the moment of writing, is the
fix.

These are completions, not chat turns: no template, no system prompt. The
runtime is being asked to continue text, which is the narrowest thing it can be
asked to do and therefore the one whose divergence is easiest to attribute.

  build_gen_prompts.py <tokenizer.json> <out.json> [--bin PATH]

`inkling_encode` is located by `--bin` or `$INKLING_ENCODE`, with no baked-in
default -- see `resolve_encoder` below for why a guessed path is worse than no
path at all.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

PROMPTS = [
    ("gen-prose", "prose",
     "The Baltic port of Bremerhaven began as a single quay, and"),
    ("gen-explain", "explanatory",
     "The reason a heavier object does not fall faster than a lighter one in a vacuum is that"),
    ("gen-arith", "arithmetic",
     "Q: A train leaves at 09:20 and arrives at 13:05. How long is the journey?\n"
     "A: Let us work it out step by step."),
    ("gen-code", "code",
     "# Return the number of vowels in a string.\ndef count_vowels(s):"),
    ("gen-list", "structured",
     "Three differences between a hash table and a balanced binary search tree:\n1."),
    ("gen-dialogue", "dialogue",
     "\"You said the gate would hold,\" she said. He looked at the river and"),
    ("gen-defn", "definitional",
     "In information theory, the entropy of a discrete random variable is"),
    ("gen-recipe", "procedural",
     "To temper chocolate without a thermometer, first"),
]


def resolve_encoder(arg):
    """`--bin`, else `$INKLING_ENCODE`, else stop and name both ways to fix it.

    There is deliberately no fallback path. A default is an opinion about
    someone else's disk, it is wrong on every machine but the one it was written
    on, and it fails in the worst way available: it looks in the guessed place
    and reports the tool missing rather than the path guessed. Same policy as
    `src/paths.rs` on the Rust side.
    """
    p = arg or os.environ.get("INKLING_ENCODE")
    if not p:
        raise SystemExit(
            "inkling_encode was not located.\n"
            "  pass --bin PATH, or set $INKLING_ENCODE\n"
            "  build it with: cargo build --release --features tokenizer "
            "--bin inkling_encode")
    if not os.path.exists(p):
        raise SystemExit(f"inkling_encode not found at {p} "
                         "(from --bin or $INKLING_ENCODE)")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("out")
    ap.add_argument("--bin", help="path to inkling_encode; "
                    "defaults to $INKLING_ENCODE, never to a baked-in path")
    args = ap.parse_args()
    args.bin = resolve_encoder(args.bin)

    items = []
    with tempfile.TemporaryDirectory() as td:
        for key, family, text in PROMPTS:
            p = os.path.join(td, key + ".ids")
            r = subprocess.run([args.bin, args.tokenizer, p], input=text.encode(),
                               capture_output=True)
            if r.returncode != 0:
                raise SystemExit(f"{key}: inkling_encode failed\n{r.stderr.decode()}")
            log = r.stdout.decode()
            # `inkling_encode` prints the decode round trip and says whether it
            # equals the input. Refusing to build an item whose ids do not
            # decode back to its prompt is the whole point of using it.
            if "== the input" not in log:
                raise SystemExit(f"{key}: round trip is not the input\n{log}")
            raw = open(p, "rb").read()
            ids = list(struct.unpack("<%dq" % (len(raw) // 8), raw))
            items.append({"key": key, "family": family, "prompt": text, "ids": ids})
            print(f"  {key:<14} {len(ids):>4} ids   {text[:52]!r}")

    json.dump({"items": items}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {len(items)} prompts, "
          f"{sum(len(i['ids']) for i in items)} prompt tokens")


if __name__ == "__main__":
    main()
