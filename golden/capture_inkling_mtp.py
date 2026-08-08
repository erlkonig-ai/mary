#!/usr/bin/env python3
"""Capture the Inkling MTP blocks on real weights — as far as a reference exists.

transformers ships the MTP CONFIG surface and no implementation:
`_keys_to_ignore_on_load_unexpected = [r"model\\.mtp\\..*"]` discards every MTP
weight on load, `mtp_layer_types` and `mtp_mlp_layer_types` are properties
nothing consumes, and there is no class anywhere in the package that uses
`input_proj` or `hidden_norm`.

So this captures exactly what CAN be checked. Each MTP layer's
`transformer_block.*` is shape-identical to an ordinary decoder layer, so it
loads into an InklingDecoderLayer and its arithmetic is checkable against the
same reference everything else was. The three wrapper tensors —
`embed_norm`, `hidden_norm`, `input_proj [hidden, 2*hidden]` — are real weights
that no reference consumes, so they get fingerprints and nothing more.

What is NOT captured, because nothing upstream defines it: how the wrapper
composes. `mtp_hidden_states_first` is True and `input_proj` takes `2*hidden`,
which says the hidden state and the embedding are concatenated on the input
side with hidden first — but that is reading a flag, not observing a
computation, and no oracle here can confirm it.

MTP layer 0 is LOCAL and layer 1 is GLOBAL (mtp local_layer_ids [0,2,4,5,6,7]),
so both attention kinds are covered.

  usage: capture_inkling_mtp.py <checkpoint dir> <out dir>
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

CKPT = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingDecoderLayer

torch.manual_seed(20260808)
torch.set_default_dtype(torch.float32)

cfgj = json.load(open(CKPT + "/config.json"))
raw = dict(cfgj["text_config"])
raw.setdefault("moe_intermediate_size", raw["intermediate_size"])
mtp_local = set(cfgj["mtp_config"]["local_layer_ids"])
n_mtp = cfgj["mtp_config"]["num_nextn_predict_layers"]

weight_map = json.load(open(CKPT + "/model.safetensors.index.json"))["weight_map"]


def get(name):
    with safe_open(CKPT + "/" + weight_map[name], framework="pt") as f:
        return f.get_tensor(name).float()


def _deint(t, dim):
    """Split an interleaved fused gate/up tensor: even rows gate, odd rows up."""
    n = t.shape[dim]
    assert n % 2 == 0, (t.shape, dim)
    idx_g = torch.arange(0, n, 2)
    idx_u = torch.arange(1, n, 2)
    return t.index_select(dim, idx_g).contiguous(), t.index_select(dim, idx_u).contiguous()


def w(name, t):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
    open(os.path.join(OUT, name), "wb").write(a.tobytes())


T = 8
H = raw["hidden_size"]
manifest = {"tokens": T, "hidden": H, "n_mtp": n_mtp,
            "mtp_local_layer_ids": sorted(mtp_local),
            "kernel": raw["sconv_kernel_size"],
            "rms_norm_eps": raw["rms_norm_eps"],
            "dense_intermediate": raw["dense_intermediate_size"],
            "d_rel": raw["d_rel"],
            "sliding_window": raw["sliding_window_size"],
            "rel_extent_global": raw["rel_extent"],
            "log_scaling_n_floor": raw["log_scaling_n_floor"],
            "log_scaling_alpha": raw["log_scaling_alpha"],
            "layers": {}}

x = torch.randn(1, T, H) * 0.05
w("mtp_x.bin", x)

for idx in (0, 1):
    is_local = idx in mtp_local
    tag = "local" if is_local else "global"
    # Build a config whose layer 0 has this MTP layer's attention kind, and is
    # dense — MTP blocks are always dense whatever dense_mlp_idx says.
    r = dict(raw)
    r["num_hidden_layers"] = 1
    r["local_layer_ids"] = [0] if is_local else []
    r["dense_mlp_idx"] = 1
    cfg = InklingTextConfig(**r)
    cfg._attn_implementation = "eager"
    assert cfg.layer_types == (["hybrid_sliding"] if is_local else ["hybrid"]), cfg.layer_types
    assert cfg.mlp_layer_types == ["dense"], cfg.mlp_layer_types

    p = f"model.mtp.layers.{idx}.transformer_block."
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
    fused = get(p + "mlp.w13_dn.weight")
    g_, u_ = _deint(fused, 0)                     # INTERLEAVED, not halved
    sd["mlp.gate_proj.weight"] = g_
    sd["mlp.up_proj.weight"] = u_
    sd["mlp.down_proj.weight"] = get(p + "mlp.w2_md.weight")
    sd["mlp.global_scale"] = get(p + "mlp.global_scale")

    layer = InklingDecoderLayer(cfg, 0)
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (sorted(missing)[:6], sorted(unexpected)[:6])

    mk = create_sliding_window_causal_mask if is_local else create_causal_mask
    mask = mk(config=cfg, inputs_embeds=x, attention_mask=None,
              past_key_values=None, position_ids=torch.arange(T).unsqueeze(0))
    with torch.no_grad():
        y = layer(x, attention_mask=mask, conv_mask=None, past_key_values=None)
    w(f"mtp_{tag}_y.bin", y)

    rel = layer.self_attn.rel_logits_proj.proj.shape
    manifest["layers"][tag] = {
        "mtp_index": idx,
        "is_local": is_local,
        "rel_proj_shape": list(rel),
        "num_heads": layer.self_attn.num_heads,
        "num_kv_heads": layer.self_attn.num_key_value_heads,
        "head_dim": layer.self_attn.head_dim,
        "rel_extent": layer.self_attn.rel_extent,
    }
    print("mtp %-6s (layer %d): heads %d kv %d dim %d, rel_extent %d, rel_proj %s, |y| max %.6g"
          % (tag, idx, layer.self_attn.num_heads, layer.self_attn.num_key_value_heads,
             layer.self_attn.head_dim, layer.self_attn.rel_extent, tuple(rel), y.abs().max()))
    del layer

# The wrapper tensors no reference consumes: fingerprints only.
wrap = {}
for idx in range(n_mtp):
    for nm in ("embed_norm.weight", "hidden_norm.weight", "input_proj.weight"):
        t = get(f"model.mtp.layers.{idx}.{nm}")
        wrap[f"{idx}.{nm}"] = {"shape": list(t.shape), "sum": float(t.sum())}
manifest["wrapper"] = wrap
ip = wrap["0.input_proj.weight"]["shape"]
manifest["input_proj_shape"] = ip
print("\nwrapper tensors present for all %d MTP layers; input_proj is %s" % (n_mtp, ip))
print("  [hidden, 2*hidden] = the embedding and hidden state concatenated on the INPUT side.")
print("  mtp_hidden_states_first=%s says hidden comes first -- a flag, not an observation."
      % cfgj["mtp_config"].get("mtp_hidden_states_first", True))
print("  NO reference implements the composition, so it is not captured and cannot be gated.")

# Are the two blocks actually different functions on this corpus?
ll, lg = manifest["layers"]["local"], manifest["layers"]["global"]
assert ll["rel_extent"] != lg["rel_extent"], "the two MTP kinds share a rel_extent"
print("\nlocal rel_extent %d vs global %d — the two kinds differ"
      % (ll["rel_extent"], lg["rel_extent"]))

json.dump(manifest, open(os.path.join(OUT, "mtp_manifest.json"), "w"), indent=1)
print("wrote oracle to", OUT)
