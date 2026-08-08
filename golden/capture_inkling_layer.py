#!/usr/bin/env python3
"""Capture oracle vectors for the Inkling MLPs and the whole decoder layer.

Two deliberate choices in the configuration, both so that a check can see what
the real checkpoints hide:

  * `moe_intermediate_size` is chosen so `2 * intermediate != hidden`. In BOTH
    released models they are equal (2048/4096 and 3072/6144), which makes the
    stacked expert matrix SQUARE and its orientation invisible to any shape
    check -- a transposed `gate_up_proj` would load happily and compute
    nonsense. Here it is 24 against 32, so the orientation is observable.
  * layers 0 and 1 are both sliding, differing ONLY in dense vs sparse MLP, so
    the layer gate isolates the MLP branch. The attention kinds are gated
    separately by capture_inkling_attn.py.

Weights are dumped straight from `named_parameters()` so the file names track
the reference's own module paths and nothing is transcribed by hand.

  usage: capture_inkling_layer.py <out dir>
"""
import json
import os
import sys

import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "./inkling-oracle"
os.makedirs(OUT, exist_ok=True)

from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import (
    InklingDecoderLayer,
    InklingMLP,
    InklingMoE,
)

torch.manual_seed(20260808)
torch.set_default_dtype(torch.float32)

T = 16
cfg = InklingTextConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=8,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=4,
    swa_head_dim=8,
    d_rel=4,
    rel_extent=9,
    sliding_window_size=5,
    local_layer_ids=[0, 1],      # both local: isolate the MLP branch
    dense_mlp_idx=1,             # layer 0 dense, layer 1 sparse
    sconv_kernel_size=4,
    rms_norm_eps=1e-6,
    log_scaling_n_floor=4,
    log_scaling_alpha=0.1,
    vocab_size=64,
    intermediate_size=64,        # dense MLP width (non-square already)
    moe_intermediate_size=12,    # 2*12 = 24 != 32, so w13 orientation is visible
    n_routed_experts=4,
    n_shared_experts=2,
    num_experts_per_tok=2,
    route_scale=8.0,
)
cfg._attn_implementation = "eager"
assert cfg.layer_types == ["hybrid_sliding", "hybrid_sliding"], cfg.layer_types
assert cfg.mlp_layer_types == ["dense", "sparse"], cfg.mlp_layer_types
assert 2 * cfg.moe_intermediate_size != cfg.hidden_size, "w13 would be square and unobservable"

H = cfg.hidden_size


def dump(name, t):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
    open(os.path.join(OUT, name), "wb").write(a.tobytes())


def dump_params(module, prefix):
    names = {}
    for n, p in module.named_parameters():
        fn = prefix + n.replace(".", "_") + ".bin"
        dump(fn, p)
        names[n] = list(p.shape)
    return names


manifest = {
    "tokens": T, "hidden": H,
    "moe_intermediate": cfg.moe_intermediate_size,
    "dense_intermediate": cfg.intermediate_size,
    "n_routed": cfg.n_routed_experts,
    "n_shared": cfg.n_shared_experts,
    "top_k": cfg.num_experts_per_tok,
    "route_scale": cfg.route_scale,
    "kernel": cfg.conv_kernel_size,
    "rms_norm_eps": cfg.rms_norm_eps,
    "heads": cfg.num_attention_heads,
    "kv_heads": cfg.swa_num_key_value_heads,
    "head_dim": cfg.swa_head_dim,
    "d_rel": cfg.d_rel,
    "rel_extent": cfg.sliding_window_size,   # local layers reach only the window
    "sliding_window": cfg.sliding_window_size,
    "log_scaling_n_floor": cfg.log_scaling_n_floor,
    "log_scaling_alpha": cfg.log_scaling_alpha,
}

x = torch.randn(T, H)
dump("lyr_x.bin", x)

# ------------------------------------------------------------- dense MLP ---
mlp = InklingMLP(cfg)
with torch.no_grad():
    for p in mlp.parameters():
        p.normal_(std=0.3)
    mlp.global_scale.fill_(1.7)          # not 1, so dropping it is visible
with torch.no_grad():
    y_mlp = mlp(x)
p_mlp = dump_params(mlp, "lyr_mlp_")
dump("lyr_mlp_y.bin", y_mlp)
print("dense mlp params:", p_mlp)
# What the answer would be if global_scale were ignored -- so the gate can
# require that the two differ and therefore that the scalar is under test.
dump("lyr_mlp_y_noscale.bin", y_mlp / 1.7)
manifest["mlp_global_scale"] = 1.7

# ------------------------------------------------------------------- MoE ---
moe = InklingMoE(cfg)
with torch.no_grad():
    for p in moe.parameters():
        p.normal_(std=0.3)
    moe.gate.global_scale.fill_(1.0)
with torch.no_grad():
    y_moe = moe(x)
p_moe = dump_params(moe, "lyr_moe_")
dump("lyr_moe_y.bin", y_moe)
print("moe params:", p_moe)

# The shared experts consume the ORIGINAL x, not the routed output. Capture the
# routed-only part so the gate can tell the two apart.
with torch.no_grad():
    _, topk_w, topk_i, gammas = moe.gate(x)
    routed_only = moe.experts(x.view(-1, H), topk_i, topk_w).view(T, H)
    shared_only = moe.shared_experts(x, gammas=gammas)
dump("lyr_moe_routed.bin", routed_only)
dump("lyr_moe_shared.bin", shared_only)
dump("lyr_moe_topk_idx.bin", topk_i.to(torch.int64).view(torch.int64))
open(os.path.join(OUT, "lyr_moe_topk_idx.bin"), "wb").write(
    np.ascontiguousarray(topk_i.cpu().numpy().astype("<i8")).tobytes())
dump("lyr_moe_topk_w.bin", topk_w)
dump("lyr_moe_gammas.bin", gammas)
print("moe routed |max| %.6g  shared |max| %.6g  sum-check %.6g"
      % (routed_only.abs().max(), shared_only.abs().max(),
         (routed_only + shared_only - y_moe).abs().max()))

# How many distinct experts actually get used. If every token picks the same
# pair, the per-expert indexing is barely exercised.
used = sorted(set(topk_i.flatten().tolist()))
manifest["moe_experts_used"] = len(used)
print("experts actually used: %d of %d %s" % (len(used), cfg.n_routed_experts, used))

# ------------------------------------------------------- decoder layers ---
mask = create_sliding_window_causal_mask(
    config=cfg, inputs_embeds=x.unsqueeze(0), attention_mask=None,
    past_key_values=None, position_ids=torch.arange(T).unsqueeze(0))
dump("lyr_mask.bin", mask.float())

for idx, tag in [(0, "dense"), (1, "sparse")]:
    layer = InklingDecoderLayer(cfg, idx)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(std=0.3)
        layer.input_layernorm.weight.normal_(mean=1.0, std=0.05)
        layer.post_attention_layernorm.weight.normal_(mean=1.0, std=0.05)
        layer.self_attn.q_norm.weight.normal_(mean=1.0, std=0.05)
        layer.self_attn.k_norm.weight.normal_(mean=1.0, std=0.05)
        if tag == "dense":
            layer.mlp.global_scale.fill_(1.3)
        else:
            layer.mlp.gate.global_scale.fill_(1.0)
    with torch.no_grad():
        y = layer(x.unsqueeze(0), attention_mask=mask, conv_mask=None, past_key_values=None)
    names = dump_params(layer, f"lyr_{tag}_")
    dump(f"lyr_{tag}_y.bin", y)
    print("\n%s layer -> %s ; params: %d tensors" % (tag, tuple(y.shape), len(names)))
    for n, s in sorted(names.items()):
        print("   %-52s %s" % (n, s))
    manifest[f"{tag}_param_count"] = len(names)

manifest["dense_global_scale"] = 1.3
json.dump(manifest, open(os.path.join(OUT, "lyr_manifest.json"), "w"), indent=1)
print("\nwrote oracle to", OUT)
