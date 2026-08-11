#!/usr/bin/env python3
"""Capture oracle vectors for InklingAttention at the REAL checkpoint config.

`capture_inkling_attn.py` deliberately shrinks the model so that every branch
engages: log scaling varies, the relative table is shorter than the sequence,
and the local and global head counts differ. That is the right corpus for
"is the arithmetic right". It is the wrong corpus for "does a backend matmul
agree", because a 32-wide dot product with 4 heads does not exercise blocking,
accumulation order, or a tensor core's input precision at all.

So this is the same module at hidden 4096 / 32 heads / 8 KV heads / head_dim
128, over a real prompt length. Two facts about the released 42-layer model are
worth stating rather than discovering:

  * `log_scaling_n_floor` is 128000, so at a hundred-odd tokens tau is exactly
    1 and log scaling is inert here. The toy capture is what tests it.
  * `rel_extent` is 1024 and `sliding_window_size` is 512, both longer than the
    sequence, so the out-of-range branch of the relative bias never fires here
    either. Again: the toy capture is what tests it.

What this one tests is the part the toy cannot: the same function at the widths
the forward actually runs, where a matmul that quietly truncates its inputs to
eleven mantissa bits shows up.

  usage: capture_inkling_attn_real.py <checkpoint dir> <out dir> [tokens]
"""
import json
import os
import sys

import numpy as np
import torch

CKPT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "./inkling-oracle"
T = int(sys.argv[3]) if len(sys.argv) > 3 else 109
os.makedirs(OUT, exist_ok=True)

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingAttention

torch.manual_seed(20260811)
torch.set_default_dtype(torch.float32)

raw = json.load(open(os.path.join(CKPT, "config.json")))["text_config"]
cfg = InklingTextConfig(**raw)
cfg._attn_implementation = "eager"

H = cfg.hidden_size
# A local layer and a global one, taken from the checkpoint's own split rather
# than assumed: layer 0 is local on this model and layer 5 is not.
local_idx = cfg.local_layer_ids[0]
global_idx = next(i for i in range(cfg.num_hidden_layers) if i not in set(cfg.local_layer_ids))
assert cfg.layer_types[local_idx] == "hybrid_sliding", cfg.layer_types[local_idx]
assert cfg.layer_types[global_idx] == "hybrid", cfg.layer_types[global_idx]

manifest = {
    "tokens": T, "hidden": H, "d_rel": cfg.d_rel,
    "rel_extent": cfg.rel_extent, "sliding_window": cfg.sliding_window_size,
    "kernel": cfg.conv_kernel_size, "rms_norm_eps": cfg.rms_norm_eps,
    "log_scaling_n_floor": cfg.log_scaling_n_floor,
    "log_scaling_alpha": cfg.log_scaling_alpha,
    "layers": {},
}


def dump(name, t):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
    open(os.path.join(OUT, name), "wb").write(a.tobytes())
    return a.size


# The activations a layer actually sees are RMS-normalized, so unit-scale input
# is the honest corpus; 0.1-scale would understate every cancellation.
x = torch.randn(1, T, H)
position_ids = torch.arange(T).unsqueeze(0)
mask_kwargs = dict(config=cfg, inputs_embeds=x, attention_mask=None,
                   past_key_values=None, position_ids=position_ids)
masks = {
    "hybrid": create_causal_mask(**mask_kwargs),
    "hybrid_sliding": create_sliding_window_causal_mask(**mask_kwargs),
}

total = dump("areal_x.bin", x)

for layer_idx, kind in [(local_idx, "hybrid_sliding"), (global_idx, "hybrid")]:
    tag = "local" if kind == "hybrid_sliding" else "global"
    attn = InklingAttention(cfg, layer_idx)
    with torch.no_grad():
        # 1/sqrt(fan_in) keeps the projections at the scale a trained layer has,
        # so the softmax is neither saturated nor flat.
        for n, p in attn.named_parameters():
            if p.dim() >= 2:
                p.normal_(std=(1.0 / p.shape[-1]) ** 0.5)
            else:
                p.normal_(std=0.02)
        attn.q_norm.weight.normal_(mean=1.0, std=0.05)
        attn.k_norm.weight.normal_(mean=1.0, std=0.05)
        attn.rel_logits_proj.proj.normal_(std=0.05)

    m = masks[kind]
    if m is None:
        raise SystemExit(f"{tag}: mask factory returned None; the oracle would not be causal")
    with torch.no_grad():
        y, _ = attn(x, attention_mask=m, conv_mask=None, past_key_values=None)

    p = f"areal_{tag}_"
    for name, t in [
        ("wq.bin", attn.q_proj.weight), ("wk.bin", attn.k_proj.weight),
        ("wv.bin", attn.v_proj.weight), ("wr.bin", attn.r_proj.weight),
        ("wo.bin", attn.o_proj.weight),
        ("k_sconv.bin", attn.k_sconv.conv1d.weight),
        ("v_sconv.bin", attn.v_sconv.conv1d.weight),
        ("q_norm.bin", attn.q_norm.weight), ("k_norm.bin", attn.k_norm.weight),
        ("rel_proj.bin", attn.rel_logits_proj.proj),
        ("y.bin", y),
    ]:
        total += dump(p + name, t)

    manifest["layers"][tag] = {
        "layer_idx": layer_idx,
        "is_sliding": kind == "hybrid_sliding",
        "num_heads": attn.num_heads,
        "num_kv_heads": attn.num_key_value_heads,
        "head_dim": attn.head_dim,
        "rel_extent": attn.rel_extent,
        "scaling": attn.scaling,
        "mask_finite_frac": float(torch.isfinite(m).float().mean()),
        "y_absmax": float(y.abs().max()),
    }
    print("%-7s layer %2d  heads %d kv %d dim %d  rel_extent %4d  scaling %.6g  |y| max %.6g"
          % (tag, layer_idx, attn.num_heads, attn.num_key_value_heads, attn.head_dim,
             attn.rel_extent, attn.scaling, y.abs().max()))

json.dump(manifest, open(os.path.join(OUT, "areal_manifest.json"), "w"), indent=1)
print("wrote %d floats (%.2f GB) to %s" % (total, total * 4 / 1e9, OUT))
