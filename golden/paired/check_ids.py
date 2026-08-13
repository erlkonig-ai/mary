#!/usr/bin/env python3
"""Do the committed ids actually say the committed prompts? Ask a second tokenizer.

`build_items.py` tokenises with `transformers.AutoTokenizer`. The runtime eats
raw ids and never sees text, so nothing downstream would notice if those two
drifted — and that exact drift has already happened in this tree: a gate
asserted one prompt while its ids file held another, and the assertion passed
because nothing ever decoded the file. `inkling_encode` exists because of it.

So this re-encodes every prompt with `inkling_encode` — the Rust
`tokenizers` executor over the checkpoint's own `tokenizer.json`, which is a
different implementation from the python one that produced the committed ids —
and compares. Two independent tokenizers agreeing on 60 prompts is a much
stronger statement than one tokenizer agreeing with itself, and it is the only
part of this harness where the two arms' inputs could silently differ.

`--mutate` appends a character to one prompt before encoding, so the check can
be watched failing.

  check_ids.py <items.json> <tokenizer.json> [--bin PATH] [--mutate]

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
    ap.add_argument("items")
    ap.add_argument("tokenizer")
    ap.add_argument("--bin", help="path to inkling_encode; "
                    "defaults to $INKLING_ENCODE, never to a baked-in path")
    ap.add_argument("--mutate", action="store_true")
    args = ap.parse_args()
    args.bin = resolve_encoder(args.bin)

    items = json.load(open(args.items))["items"]
    bad = []
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "p.ids")
        for n, it in enumerate(items):
            text = it["prompt"] + ("." if args.mutate and n == 0 else "")
            r = subprocess.run([args.bin, args.tokenizer, out], input=text.encode(),
                               capture_output=True)
            if r.returncode != 0:
                raise SystemExit(f"{it['key']}: inkling_encode failed\n{r.stderr.decode()}")
            raw = open(out, "rb").read()
            got = list(struct.unpack("<%dq" % (len(raw) // 8), raw))
            if got != list(it["ids"]):
                bad.append((it["key"], len(it["ids"]), len(got)))

    print(f"{len(items) - len(bad)}/{len(items)} prompts encode identically under "
          f"inkling_encode (Rust tokenizers) and AutoTokenizer (python)")
    for key, a, b in bad:
        print(f"  MISMATCH {key}: {a} committed ids vs {b} re-encoded")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
