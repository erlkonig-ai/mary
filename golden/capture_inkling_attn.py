#!/usr/bin/env python3
"""Capture oracle vectors for InklingAttention — one LOCAL and one GLOBAL layer.

Both are captured because they are different functions, not two settings of one:
they differ in head counts (`swa_*` vs the global fields), in how far the
relative-position table reaches (`sliding_window_size` vs `rel_extent`), in
whether a sliding-window mask applies, and in whether log scaling runs at all.
A gate on either alone cannot see the difference.

The configuration is small but chosen so that every branch actually engages,
because the real values would make several of them inert at this scale:

  * `log_scaling_n_floor` is small here. With the checkpoint's 128000 and a
    short sequence, tau = 1 + alpha*log(clamp((n+1)/n_floor, min=1)) is exactly
    1 and log scaling is a no-op -- a gate would "cover" it while testing
    nothing. The manifest records the tau spread so the gate can insist it
    varies.
  * `rel_extent` is SHORTER than the sequence, so the `distance >= rel_extent`
    branch fires. With the real 1024 and 16 tokens it never would.
  * `swa_num_key_value_heads` differs from `num_key_value_heads`, mirroring the
    66-layer model where they are 16 and 8. Where they agree -- as in the
    42-layer model -- the two readings are indistinguishable.

  usage: capture_inkling_attn.py <out dir>
"""
import json
import math
import os
import sys

import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "./inkling-oracle"
os.makedirs(OUT, exist_ok=True)

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingAttention

torch.manual_seed(20260808)
torch.set_default_dtype(torch.float32)

T = 16
cfg = InklingTextConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,      # global: 2 KV heads -> 2 GQA groups
    head_dim=8,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=4,  # local: 4 KV heads -> 1 group. Deliberately different.
    swa_head_dim=8,
    d_rel=4,
    rel_extent=9,               # < T, so the out-of-range branch fires
    sliding_window_size=5,      # < T, so the window mask bites
    local_layer_ids=[0],        # layer 0 local, layer 1 global
    sconv_kernel_size=4,
    rms_norm_eps=1e-6,
    log_scaling_n_floor=4,      # small, so tau actually varies
    log_scaling_alpha=0.1,
    vocab_size=64,
    intermediate_size=64,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=2,
    num_experts_per_tok=2,
    route_scale=8.0,
)
cfg._attn_implementation = "eager"
assert cfg.layer_types == ["hybrid_sliding", "hybrid"], cfg.layer_types

H = cfg.hidden_size
manifest = {"tokens": T, "hidden": H, "d_rel": cfg.d_rel,
            "rel_extent": cfg.rel_extent, "sliding_window": cfg.sliding_window_size,
            "kernel": cfg.conv_kernel_size, "rms_norm_eps": cfg.rms_norm_eps,
            "log_scaling_n_floor": cfg.log_scaling_n_floor,
            "log_scaling_alpha": cfg.log_scaling_alpha,
            "heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "swa_heads": cfg.swa_num_attention_heads,
            "swa_kv_heads": cfg.swa_num_key_value_heads,
            "swa_head_dim": cfg.swa_head_dim,
            "layers": {}}


def dump(name, t):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
    open(os.path.join(OUT, name), "wb").write(a.tobytes())
    return a


x = torch.randn(1, T, H)
position_ids = torch.arange(T).unsqueeze(0)
mask_kwargs = dict(config=cfg, inputs_embeds=x, attention_mask=None,
                   past_key_values=None, position_ids=position_ids)
masks = {
    "hybrid": create_causal_mask(**mask_kwargs),
    "hybrid_sliding": create_sliding_window_causal_mask(**mask_kwargs),
}

dump("attn_x.bin", x)

for layer_idx, kind in [(0, "hybrid_sliding"), (1, "hybrid")]:
    tag = "local" if kind == "hybrid_sliding" else "global"
    attn = InklingAttention(cfg, layer_idx)
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(std=0.3)
        # Norm gains near 1 so QK-norm is a perturbation, not an erasure.
        attn.q_norm.weight.normal_(mean=1.0, std=0.05)
        attn.k_norm.weight.normal_(mean=1.0, std=0.05)

    m = masks[kind]
    if m is None:
        raise SystemExit(f"{tag}: mask factory returned None; the oracle would not be causal")
    with torch.no_grad():
        y, w = attn(x, attention_mask=m, conv_mask=None, past_key_values=None)

    p = f"attn_{tag}_"
    dump(p + "wq.bin", attn.q_proj.weight)
    dump(p + "wk.bin", attn.k_proj.weight)
    dump(p + "wv.bin", attn.v_proj.weight)
    dump(p + "wr.bin", attn.r_proj.weight)
    dump(p + "wo.bin", attn.o_proj.weight)
    dump(p + "k_sconv.bin", attn.k_sconv.conv1d.weight)
    dump(p + "v_sconv.bin", attn.v_sconv.conv1d.weight)
    dump(p + "q_norm.bin", attn.q_norm.weight)
    dump(p + "k_norm.bin", attn.k_norm.weight)
    dump(p + "rel_proj.bin", attn.rel_logits_proj.proj)
    dump(p + "mask.bin", m.float())
    dump(p + "y.bin", y)

    # tau, so the gate can require that log scaling is actually doing something
    # on the global layer and is absent on the local one.
    q_pos = torch.arange(T).float()
    tau = 1.0 + cfg.log_scaling_alpha * torch.log(
        (( q_pos + 1) / cfg.log_scaling_n_floor).clamp(min=1.0))
    applies = (kind == "hybrid")
    manifest["layers"][tag] = {
        "layer_idx": layer_idx,
        "is_sliding": kind == "hybrid_sliding",
        "num_heads": attn.num_heads,
        "num_kv_heads": attn.num_key_value_heads,
        "head_dim": attn.head_dim,
        "rel_extent": attn.rel_extent,
        "scaling": attn.scaling,
        "log_scaling_applies": applies,
        "tau_min": float(tau.min()) if applies else 1.0,
        "tau_max": float(tau.max()) if applies else 1.0,
        "mask_shape": list(m.shape),
        "mask_finite_frac": float(torch.isfinite(m).float().mean()),
    }
    print("%-7s layer %d  heads %d kv %d dim %d  rel_extent %d  scaling %.6g  mask %s"
          % (tag, layer_idx, attn.num_heads, attn.num_key_value_heads, attn.head_dim,
             attn.rel_extent, attn.scaling, tuple(m.shape)))
    print("          tau in [%.6g, %.6g]  applies=%s   |y| max %.6g"
          % (manifest["layers"][tag]["tau_min"], manifest["layers"][tag]["tau_max"],
             applies, y.abs().max()))

# Sanity the two layers really are different functions.
lg, ll = manifest["layers"]["global"], manifest["layers"]["local"]
assert lg["num_kv_heads"] != ll["num_kv_heads"], "the two layers share a head config"
assert lg["rel_extent"] != ll["rel_extent"], "the two layers share a rel_extent"
assert lg["log_scaling_applies"] != ll["log_scaling_applies"]
assert lg["tau_max"] > 1.0 + 1e-6, "log scaling is inert -- the global gate would not test it"
print("\nlocal vs global differ in kv heads (%d vs %d), rel_extent (%d vs %d), log scaling (%s vs %s)"
      % (ll["num_kv_heads"], lg["num_kv_heads"], ll["rel_extent"], lg["rel_extent"],
         ll["log_scaling_applies"], lg["log_scaling_applies"]))

json.dump(manifest, open(os.path.join(OUT, "attn_manifest.json"), "w"), indent=1)
print("wrote oracle to", OUT)
