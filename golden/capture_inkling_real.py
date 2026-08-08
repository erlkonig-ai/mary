#!/usr/bin/env python3
"""Run ONE REAL layer of the released checkpoint through the reference module.

The released weights use the original TML names (`wq_du`, `w13_dn`,
`shared_w13_weight`) while `transformers` uses module names (`q_proj`,
`gate_proj`), so the mapping has to be authored. `load_state_dict(strict=True)`
makes its TOTALITY machine-checked -- every target parameter filled, nothing
left over -- which is the part that can be checked mechanically.

What CANNOT be checked by comparing two of my own implementations: whether
`w13` splits as [gate; up] or [up; gate]. Both lanes would share the mistake.
The split follows the LLaMA convention that w1 is the gate and w3 the up
projection, and the MoE path corroborates it -- the reference chunks the fused
`gate_up_proj` output as (gate, up) along the output dimension, and weight row
i produces output i, so gate occupies the first half of the rows. It is
recorded here as an assumption rather than a verified fact. What would settle
it is a full forward producing coherent text, which needs more memory than one
Spark has.

Weights are not dumped -- they are gigabytes and Rust reads them from the
checkpoint itself, which is the point. Instead each mapped tensor gets a
compact fingerprint so a loading error is localised to a tensor rather than
smeared across the layer output.

  usage: capture_inkling_real.py <checkpoint dir> <out dir> [layer_idx]
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

CKPT = sys.argv[1]
OUT = sys.argv[2]
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(OUT, exist_ok=True)

from transformers.masking_utils import create_sliding_window_causal_mask, create_causal_mask
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingDecoderLayer

torch.manual_seed(20260808)
torch.set_default_dtype(torch.float32)

raw = json.load(open(CKPT + "/config.json"))["text_config"]
cfg = InklingTextConfig(**raw)
cfg._attn_implementation = "eager"
H = cfg.hidden_size
T = 16

weight_map = json.load(open(CKPT + "/model.safetensors.index.json"))["weight_map"]


def get(name):
    with safe_open(CKPT + "/" + weight_map[name], framework="pt") as f:
        return f.get_tensor(name).float()


pfx = f"model.llm.layers.{LAYER}."
is_dense = cfg.mlp_layer_types[LAYER] == "dense"
is_sliding = cfg.layer_types[LAYER] == "hybrid_sliding"
print(f"layer {LAYER}: mlp={'dense' if is_dense else 'sparse'}, "
      f"attn={'sliding' if is_sliding else 'global'}")

sd = {
    "input_layernorm.weight": get(pfx + "attn_norm.weight"),
    "post_attention_layernorm.weight": get(pfx + "mlp_norm.weight"),
    "attn_sconv.conv1d.weight": get(pfx + "attn_sconv.weight"),
    "mlp_sconv.conv1d.weight": get(pfx + "mlp_sconv.weight"),
    "self_attn.q_proj.weight": get(pfx + "attn.wq_du.weight"),
    "self_attn.k_proj.weight": get(pfx + "attn.wk_dv.weight"),
    "self_attn.v_proj.weight": get(pfx + "attn.wv_dv.weight"),
    "self_attn.r_proj.weight": get(pfx + "attn.wr_du.weight"),
    "self_attn.o_proj.weight": get(pfx + "attn.wo_ud.weight"),
    "self_attn.q_norm.weight": get(pfx + "attn.q_norm.weight"),
    "self_attn.k_norm.weight": get(pfx + "attn.k_norm.weight"),
    "self_attn.k_sconv.conv1d.weight": get(pfx + "attn.k_sconv.weight"),
    "self_attn.v_sconv.conv1d.weight": get(pfx + "attn.v_sconv.weight"),
    "self_attn.rel_logits_proj.proj": get(pfx + "attn.rel_logits_proj.proj"),
}

if is_dense:
    w13 = get(pfx + "mlp.w13_dn.weight")          # [2 * dense_inter, hidden]
    half = w13.shape[0] // 2
    sd["mlp.gate_proj.weight"] = w13[:half].contiguous()   # w1
    sd["mlp.up_proj.weight"] = w13[half:].contiguous()     # w3
    sd["mlp.down_proj.weight"] = get(pfx + "mlp.w2_md.weight")
    sd["mlp.global_scale"] = get(pfx + "mlp.global_scale")
else:
    raise SystemExit("sparse layers not handled by this capture yet")

layer = InklingDecoderLayer(cfg, LAYER)
missing, unexpected = layer.load_state_dict(sd, strict=False)
print("  state_dict: %d mapped, %d missing, %d unexpected"
      % (len(sd), len(missing), len(unexpected)))
if missing or unexpected:
    print("  missing   :", sorted(missing)[:8])
    print("  unexpected:", sorted(unexpected)[:8])
    raise SystemExit("mapping is not total -- refusing to emit an oracle from a partial load")

x = torch.randn(1, T, H) * 0.05      # activations at a plausible scale
mk = create_sliding_window_causal_mask if is_sliding else create_causal_mask
mask = mk(config=cfg, inputs_embeds=x, attention_mask=None,
          past_key_values=None, position_ids=torch.arange(T).unsqueeze(0))
with torch.no_grad():
    y = layer(x, attention_mask=mask, conv_mask=None, past_key_values=None)


def w(name, t):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
    open(os.path.join(OUT, name), "wb").write(a.tobytes())


w("real_x.bin", x)
w("real_y.bin", y)

# Compact per-tensor fingerprints: a loading error lands on one line instead of
# smearing across the output.
fps = {}
for k, v in sorted(sd.items()):
    f = v.flatten()
    fps[k] = {
        "shape": list(v.shape),
        "sum": float(f.sum()),
        "sumsq": float((f.double() ** 2).sum()),
        "min": float(f.min()), "max": float(f.max()),
        "first4": [float(z) for z in f[:4]],
    }

manifest = {
    "layer": LAYER, "tokens": T, "hidden": H,
    "is_dense": is_dense, "is_sliding": is_sliding,
    "heads": cfg.swa_num_attention_heads if is_sliding else cfg.num_attention_heads,
    "kv_heads": cfg.swa_num_key_value_heads if is_sliding else cfg.num_key_value_heads,
    "head_dim": cfg.swa_head_dim if is_sliding else cfg.head_dim,
    "d_rel": cfg.d_rel,
    "rel_extent": cfg.sliding_window_size if is_sliding else cfg.rel_extent,
    "sliding_window": cfg.sliding_window_size,
    "kernel": cfg.conv_kernel_size,
    "rms_norm_eps": cfg.rms_norm_eps,
    "dense_intermediate": cfg.intermediate_size,
    "log_scaling_n_floor": cfg.log_scaling_n_floor,
    "log_scaling_alpha": cfg.log_scaling_alpha,
    "w13_split": "rows [0:half] = gate (w1), [half:] = up (w3)  -- ASSUMPTION",
    "fingerprints": fps,
}
json.dump(manifest, open(os.path.join(OUT, "real_manifest.json"), "w"), indent=1)

print("  x %s -> y %s   |y| max %.6g" % (tuple(x.shape), tuple(y.shape), y.abs().max()))
print("  tau at this length: 1.0 (n_floor %d, seq %d) -- log scaling INERT here"
      % (cfg.log_scaling_n_floor, T))
for k in sorted(fps):
    print("   %-40s %-18s sum %+.6e" % (k, fps[k]["shape"], fps[k]["sum"]))
print("wrote oracle to", OUT)
