#!/usr/bin/env python3
"""The BF16 reference side of the paired capability comparison: a whole forward
of the unquantised checkpoint, one layer at a time, on a box that cannot hold it.

# Why this file exists at all

`transformers` loads this checkpoint natively, and cannot run it here: the BF16
release is 531.9 GB of tensors against 119 GiB of unified memory, so
`from_pretrained` either OOMs or falls into accelerate's disk offload, which
wants another 531.9 GB of scratch that this filesystem does not have (364 GB
free). What it CAN do is build one `InklingDecoderLayer` at a time — which is
exactly what `capture_inkling_real.py` has been doing for a single layer since
the port started. This is that, forty-two times, carrying the residual stream
across.

So the peak resident set is one layer (12.9 GB for a sparse layer: 8.6 GB of
`w13` plus 4.3 GB of `w2`) plus the embedding and unembedding tables, and the
cost of a forward is one pass over the checkpoint from the SSD. Measured here
at 5.1 GB/s (`dd iflag=direct`), that floor is ~104 s, and the run reports what
it actually paid.

# The batch is the whole point

A full stream is per FORWARD PASS, not per token and not per prompt. One pass
over 531.9 GB prefills one prompt or forty for the same money, so every prompt
in a set goes through together and the reference costs one stream per set
rather than one per item. That is the difference between a 3-minute experiment
and a 2-hour one, and it is why the harness above this asks its questions in a
form whose answer is the FIRST generated token: a first token needs a prefill
and no decode step, and each decode step would be another full stream.

# Padding is on the right, and that is load-bearing

Prompts are right-padded to a common length and each item's logits are read at
its own last real position. Every path in this model is causal — attention
under `create_causal_mask` / `create_sliding_window_causal_mask`, and the short
convolutions look backwards over `sconv_kernel_size` — so a token can never see
a position after it, and the pads are all after every real token. Batching
therefore cannot change any item's answer.

That is an argument, not a measurement, so `--selfcheck` measures it: it runs
the first item alone and again inside the full batch and reports the max
absolute difference of the two last-position logit rows. Run it once when the
prompt set changes.

  inkling_bf16_stream.py <ckpt> <items.json> <out.json> [--layers N] [--selfcheck]

`items.json` is `{"items": [{"key": ..., "ids": [...], "option_ids": [...]}]}`;
`option_ids` is optional and names token ids whose logits are recorded exactly
(the answer letters), on top of the top-32 that is always recorded.

`--layers N` stops after N layers and writes no logits. It is the cost probe:
the per-layer seconds it prints are what the full run will pay, times 42.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch


_ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
              "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8}


class Shard:
    """A safetensors shard read with `pread`, not `mmap`.

    `safe_open(...).get_tensor()` memory-maps and lets page faults pull the
    bytes in, which on this box runs an 8.6 GB expert stack off the SSD at
    0.19 GiB/s — 42 s a layer, 29 minutes a forward, and 96% of it waiting.
    The same file read sequentially does 5.1 GiB/s (`dd iflag=direct`), so the
    gap is fault granularity, not the device.

    The safetensors container makes the direct read trivial: 8 bytes of
    little-endian header length, that many bytes of JSON naming each tensor's
    dtype, shape and byte range, then the payload. `tensor()` preads the range
    into one buffer. `first()` is the same read for a whole shard, which is
    what a layer's worth of expert weight actually is.
    """

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        n = int.from_bytes(os.pread(self.fd, 8, 0), "little")
        self.header = json.loads(os.pread(self.fd, n, 8))
        self.base = 8 + n

    def tensor(self, name):
        e = self.header[name]
        a, b = e["data_offsets"]
        nbytes = b - a
        buf = torch.empty(nbytes, dtype=torch.uint8)
        mv = memoryview(buf.numpy())
        off, done = self.base + a, 0
        while done < nbytes:
            k = os.preadv(self.fd, [mv[done: done + (1 << 30)]], off + done)
            if k == 0:
                raise EOFError(f"{self.path}: short read for {name}")
            done += k
        return buf.view(_ST_DTYPES[e["dtype"]]).view(*e["shape"])


def io_read_bytes():
    """Bytes this process pulled off the block device — page-cache hits are free.

    The same instrument `inkling_forward` uses, for the same reason: on a box
    whose page cache is 107 GB against a 531.9 GB checkpoint, "how much did we
    read" and "how much did we ask for" are different numbers and only the
    first one costs time.
    """
    try:
        for line in open("/proc/self/io"):
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=0, help="stop after N layers (cost probe)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="also run item 0 alone and report the batching delta")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mutate", default="", choices=["", "halved-w13"],
                    help="deliberately break the reference, to watch the harness catch it")
    args = ap.parse_args()

    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
        create_sliding_window_causal_mask,
    )
    from transformers.models.inkling.configuration_inkling import InklingTextConfig
    from transformers.models.inkling.modeling_inkling import (
        InklingDecoderLayer,
        InklingRMSNorm,
    )

    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    DT = torch.bfloat16

    raw = json.load(open(args.ckpt + "/config.json"))["text_config"]
    # See capture_inkling_real.py: the checkpoint's `intermediate_size` IS the
    # per-expert width, and transformers would otherwise fall back to a default
    # that is only right for the 66-layer release.
    raw.setdefault("moe_intermediate_size", raw["intermediate_size"])
    cfg = InklingTextConfig(**raw)
    cfg._attn_implementation = "eager"

    weight_map = json.load(open(args.ckpt + "/model.safetensors.index.json"))["weight_map"]
    _open = {}

    def get(name, dtype=DT, device=dev):
        shard = weight_map[name]
        f = _open.get(shard)
        if f is None:
            f = _open[shard] = Shard(args.ckpt + "/" + shard)
        return f.tensor(name).to(device=device, dtype=dtype)

    def deint(t, dim):
        """Even rows gate, odd rows up — the interleaving settled in 2026-08."""
        n = t.shape[dim]
        assert n % 2 == 0, (t.shape, dim)
        return (t.index_select(dim, torch.arange(0, n, 2, device=t.device)).contiguous(),
                t.index_select(dim, torch.arange(1, n, 2, device=t.device)).contiguous())

    def split_w13(t, dim):
        """The gate/up split, or — under `--mutate halved-w13` — the wrong one.

        `halved-w13` is the reading this tree ACTUALLY held until 2026-08: rows
        [0:inter] as gate and [inter:] as up, rather than even/odd. It is
        shape-identical and numerically different, which is precisely why it
        survived so long, and it is the cheapest real fault available: no code
        path changes, no shape changes, nothing raises. If the harness cannot
        tell this apart from a good run then it cannot tell anything apart, and
        a check nobody has watched fail is not evidence.
        """
        if args.mutate == "halved-w13":
            n = t.shape[dim]
            return (t.narrow(dim, 0, n // 2).contiguous(),
                    t.narrow(dim, n // 2, n // 2).contiguous())
        return deint(t, dim)

    def layer_state(i):
        p = f"model.llm.layers.{i}."
        sd = {
            "input_layernorm.weight": get(p + "attn_norm.weight"),
            "post_attention_layernorm.weight": get(p + "mlp_norm.weight"),
            "attn_sconv.conv1d.weight": get(p + "attn_sconv.weight"),
            "mlp_sconv.conv1d.weight": get(p + "mlp_sconv.weight"),
            "self_attn.q_proj.weight": get(p + "attn.wq_du.weight"),
            "self_attn.k_proj.weight": get(p + "attn.wk_dv.weight"),
            "self_attn.v_proj.weight": get(p + "attn.wv_dv.weight"),
            "self_attn.r_proj.weight": get(p + "attn.wr_du.weight"),
            "self_attn.o_proj.weight": get(p + "attn.wo_ud.weight"),
            "self_attn.q_norm.weight": get(p + "attn.q_norm.weight"),
            "self_attn.k_norm.weight": get(p + "attn.k_norm.weight"),
            "self_attn.k_sconv.conv1d.weight": get(p + "attn.k_sconv.weight"),
            "self_attn.v_sconv.conv1d.weight": get(p + "attn.v_sconv.weight"),
            "self_attn.rel_logits_proj.proj": get(p + "attn.rel_logits_proj.proj"),
        }
        if cfg.mlp_layer_types[i] == "dense":
            g, u = deint(get(p + "mlp.w13_dn.weight"), 0)
            sd["mlp.gate_proj.weight"] = g
            sd["mlp.up_proj.weight"] = u
            sd["mlp.down_proj.weight"] = get(p + "mlp.w2_md.weight")
            sd["mlp.global_scale"] = get(p + "mlp.global_scale")
        else:
            sd["mlp.gate.weight"] = get(p + "mlp.gate.weight")
            sd["mlp.gate.e_score_correction_bias"] = get(p + "mlp.gate.bias")
            sd["mlp.gate.global_scale"] = get(p + "mlp.gate.global_scale")
            g, u = split_w13(get(p + "mlp.experts.w13_weight"), 1)
            sd["mlp.experts.gate_up_proj"] = torch.cat([g, u], dim=1).contiguous()
            del g, u
            sd["mlp.experts.down_proj"] = get(p + "mlp.experts.w2_weight")
            sg, su = split_w13(get(p + "mlp.shared_experts.shared_w13_weight"), 1)
            sd["mlp.shared_experts.gate_proj"] = sg
            sd["mlp.shared_experts.up_proj"] = su
            sd["mlp.shared_experts.down_proj"] = get(p + "mlp.shared_experts.shared_w2_weight")
        return sd

    # The hand-rolled reader is checked against the library one, on a tensor
    # small enough that the slow path costs nothing. A reader that is fast and
    # wrong would poison every number downstream, and nothing else in this file
    # would notice.
    from safetensors import safe_open
    probe_name = "model.llm.layers.0.attn_norm.weight"
    with safe_open(args.ckpt + "/" + weight_map[probe_name], framework="pt") as f:
        ref = f.get_tensor(probe_name)
    mine = Shard(args.ckpt + "/" + weight_map[probe_name]).tensor(probe_name)
    assert ref.dtype == mine.dtype and ref.shape == mine.shape and torch.equal(ref, mine), \
        "the pread reader disagrees with safetensors"
    print(f"reader check: {probe_name} identical to safetensors ({tuple(ref.shape)}, {ref.dtype})",
          flush=True)

    items = json.load(open(args.items))["items"]
    print(f"items: {len(items)}", flush=True)

    def run(batch, tag, n_layers):
        """One forward over `batch` (a list of items). Returns [B, vocab] logits."""
        lens = [len(it["ids"]) for it in batch]
        T = max(lens)
        B = len(batch)
        pad = 0
        ids = torch.full((B, T), pad, dtype=torch.long, device=dev)
        amask = torch.zeros((B, T), dtype=torch.long, device=dev)
        for b, it in enumerate(batch):
            ids[b, : lens[b]] = torch.tensor(it["ids"], dtype=torch.long, device=dev)
            amask[b, : lens[b]] = 1

        embed_w = get("model.llm.embed.weight")
        h = torch.nn.functional.embedding(ids, embed_w)
        del embed_w
        enorm = InklingRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev, DT)
        enorm.weight.data = get("model.llm.embed_norm.weight")
        h = enorm(h)
        del enorm

        pos = torch.arange(T, device=dev).unsqueeze(0)
        mk = {"config": cfg, "inputs_embeds": h, "attention_mask": amask,
              "past_key_values": None, "position_ids": pos}
        masks = {
            "full_attention": create_causal_mask(**mk),
            "sliding_attention": create_sliding_window_causal_mask(**mk),
            "linear_attention": create_recurrent_attention_mask(**mk),
        }

        t0 = time.time()
        io0 = io_read_bytes()
        upto = n_layers or cfg.num_hidden_layers
        for i in range(upto):
            tl = time.time()
            sd = layer_state(i)
            tload = time.time() - tl
            with torch.device("meta"):
                layer = InklingDecoderLayer(cfg, i)
            layer.load_state_dict(sd, strict=True, assign=True)
            del sd
            kind = "full_attention" if cfg.layer_types[i] == "hybrid" else "sliding_attention"
            h = layer(h, attention_mask=masks[kind], conv_mask=masks["linear_attention"],
                      past_key_values=None)
            # Back to BF16 at the residual. `InklingMoE` hands back float32 --
            # its router scores and shared-expert gammas are computed there --
            # and a float32 residual meets the next layer's BF16 `q_proj` and
            # stops the run. A uniformly-BF16 model rounds at exactly this
            # point, so this IS the reference lane rather than a concession to
            # it: `from_pretrained(dtype=bfloat16)` would have cast the same
            # tensor at the same place.
            h = h.to(DT)
            del layer
            torch.cuda.synchronize() if dev.type == "cuda" else None
            torch.cuda.empty_cache() if dev.type == "cuda" else None
            print(f"  [{tag}] layer {i:2d} {cfg.mlp_layer_types[i]:6s}/{cfg.layer_types[i]:14s} "
                  f"load {tload:6.2f}s  total {time.time() - tl:6.2f}s  "
                  f"cum {time.time() - t0:7.1f}s  read {(io_read_bytes() - io0) / 2**30:8.1f} GiB",
                  flush=True)
        secs = time.time() - t0
        gib = (io_read_bytes() - io0) / 2**30
        print(f"  [{tag}] {upto} layers in {secs:.1f}s, {gib:.1f} GiB off the device "
              f"({gib / max(secs, 1e-9):.2f} GiB/s)", flush=True)
        if n_layers:
            return None, secs, gib

        fnorm = InklingRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev, DT)
        fnorm.weight.data = get("model.llm.norm.weight")
        h = fnorm(h)
        # Only the last REAL position of each item is ever read, so the [B, T,
        # 201024] logit tensor is never built: 40 x 256 x 201024 floats would be
        # 4.1 GB for 40 rows we want.
        last = torch.stack([h[b, lens[b] - 1] for b in range(B)]).to(DT)
        last = last / cfg.logits_mup_width_multiplier
        unembed = get("model.llm.unembed.weight")
        logits = torch.nn.functional.linear(last, unembed).float()
        del unembed
        if cfg.unpadded_vocab_size and cfg.unpadded_vocab_size < logits.shape[-1]:
            logits = logits[..., : cfg.unpadded_vocab_size]
        return logits, secs, gib

    t_all = time.time()
    if args.layers:
        run(items, "probe", args.layers)
        print(f"cost probe only; extrapolated full stack "
              f"{(time.time() - t_all) / args.layers * cfg.num_hidden_layers:.0f}s", flush=True)
        return

    logits, secs, gib = run(items, "batch", 0)

    out = {
        "checkpoint": args.ckpt,
        "n_items": len(items),
        "seconds": secs,
        "gib_read": gib,
        "dtype": "bfloat16",
        "attn_implementation": cfg._attn_implementation,
        "mutate": args.mutate,
        "results": {},
    }
    for b, it in enumerate(items):
        row = logits[b]
        vals, idx = torch.topk(row, args.topk)
        r = {
            "n_ids": len(it["ids"]),
            # The cache is keyed by the PROMPT, not by the item name. A set
            # whose wording is edited keeps its keys, and a reference costing
            # nine minutes a pass is exactly the artefact somebody will reuse
            # without re-running it. The scorer refuses a mismatch.
            "ids_sha256": hashlib.sha256(
                b"".join(int(i).to_bytes(8, "little") for i in it["ids"])).hexdigest(),
            "top_ids": [int(x) for x in idx.tolist()],
            "top_logits": [float(x) for x in vals.tolist()],
            "argmax": int(idx[0]),
        }
        if it.get("option_ids"):
            r["option_ids"] = list(it["option_ids"])
            r["option_logits"] = [float(row[i]) for i in it["option_ids"]]
        out["results"][it["key"]] = r

    # Written BEFORE the optional selfcheck, which is a second full stream: the
    # expensive artefact should not be hostage to an extra nine minutes of a
    # check that only validates the batching.
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t_all:.1f}s so far)", flush=True)

    if args.selfcheck:
        solo, _, _ = run(items[:1], "solo", 0)
        d = (solo[0] - logits[0]).abs().max().item()
        out["selfcheck_max_abs_logit_delta"] = d
        print(f"selfcheck: item {items[0]['key']!r} alone vs in a batch of "
              f"{len(items)}: max |dlogit| = {d:.3e}", flush=True)

        json.dump(out, open(args.out, "w"), indent=1)
    print(f"done in {time.time() - t_all:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
