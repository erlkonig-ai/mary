#!/usr/bin/env python3
"""Chained-input parity: feed the reference mary's ACTUAL hidden state.

Every per-layer gate so far fed a layer random inputs. This feeds layer k the
value mary really produced at layer k-1 and compares layer k's output, one
layer in torch at a time. It answers a question the parity gates cannot: does
the stack diverge somewhere, and if so, where first.

  usage: chain_check.py <ckpt> <trace dir> <layer> [<layer> ...]
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

CKPT, TRACE = sys.argv[1], sys.argv[2]
LAYERS = [int(a) for a in sys.argv[3:]] or [0]

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingDecoderLayer

torch.set_default_dtype(torch.float32)
cfgj = json.load(open(CKPT + "/config.json"))
raw = dict(cfgj["text_config"])
raw.setdefault("moe_intermediate_size", raw["intermediate_size"])
H = raw["hidden_size"]
weight_map = json.load(open(CKPT + "/model.safetensors.index.json"))["weight_map"]

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def get(name):
    with safe_open(CKPT + "/" + weight_map[name], framework="pt") as f:
        return f.get_tensor(name).float()


def get_expert(name):
    if name + ".scale" not in weight_map:
        return get(name)
    with safe_open(CKPT + "/" + weight_map[name], framework="pt") as f:
        codes = f.get_tensor(name).numpy().astype(np.uint8)
    scale = get(name + ".scale").numpy()
    scale2 = get(name + ".scale2").numpy()
    lo, hi = codes & 0x0F, (codes >> 4) & 0x0F
    vals = np.empty(codes.shape[:-1] + (codes.shape[-1] * 2,), dtype=np.float32)
    vals[..., 0::2] = FP4[lo]
    vals[..., 1::2] = FP4[hi]
    ns = scale.shape[-1]
    g = vals.shape[-1] // ns
    deq = vals.reshape(*vals.shape[:-1], ns, g) * scale[..., None]
    return torch.from_numpy(deq.reshape(*vals.shape) * scale2[:, None, None])


def load(path):
    a = np.frombuffer(open(path, "rb").read(), dtype="<f4")
    return torch.from_numpy(a.reshape(-1, H).copy())


for L in LAYERS:
    prev = os.path.join(TRACE, "h_embed.bin" if L == 0 else f"h_after_{L-1:02}.bin")
    cur = os.path.join(TRACE, f"h_after_{L:02}.bin")
    x = load(prev).unsqueeze(0)
    mine = load(cur)
    T = x.shape[1]

    is_dense = L < raw["dense_mlp_idx"]
    is_local = L in set(raw["local_layer_ids"])
    r = dict(raw)
    r["num_hidden_layers"] = 1
    r["local_layer_ids"] = [0] if is_local else []
    r["dense_mlp_idx"] = 1 if is_dense else 0
    cfg = InklingTextConfig(**r)
    cfg._attn_implementation = "eager"
    assert cfg.mlp_layer_types == (["dense"] if is_dense else ["sparse"])

    p = f"model.llm.layers.{L}."
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
    if is_dense:
        fused = get(p + "mlp.w13_dn.weight")
        half = fused.shape[0] // 2
        sd["mlp.gate_proj.weight"] = fused[:half].contiguous()
        sd["mlp.up_proj.weight"] = fused[half:].contiguous()
        sd["mlp.down_proj.weight"] = get(p + "mlp.w2_md.weight")
        sd["mlp.global_scale"] = get(p + "mlp.global_scale")
    else:
        inter = cfg.moe_intermediate_size
        sd["mlp.gate.weight"] = get(p + "mlp.gate.weight")
        sd["mlp.gate.e_score_correction_bias"] = get(p + "mlp.gate.bias")
        sd["mlp.gate.global_scale"] = get(p + "mlp.gate.global_scale")
        sd["mlp.experts.gate_up_proj"] = get_expert(p + "mlp.experts.w13_weight")
        sd["mlp.experts.down_proj"] = get_expert(p + "mlp.experts.w2_weight")
        sw = get(p + "mlp.shared_experts.shared_w13_weight")
        sd["mlp.shared_experts.gate_proj"] = sw[:, :inter].contiguous()
        sd["mlp.shared_experts.up_proj"] = sw[:, inter:].contiguous()
        sd["mlp.shared_experts.down_proj"] = get(p + "mlp.shared_experts.shared_w2_weight")

    layer = InklingDecoderLayer(cfg, 0)
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (sorted(missing)[:5], sorted(unexpected)[:5])

    mk = create_sliding_window_causal_mask if is_local else create_causal_mask
    mask = mk(config=cfg, inputs_embeds=x, attention_mask=None,
              past_key_values=None, position_ids=torch.arange(T).unsqueeze(0))
    with torch.no_grad():
        y = layer(x, attention_mask=mask, conv_mask=None, past_key_values=None)[0]

    d = (y - mine).abs()
    scale = y.abs().max()
    print("layer %2d [%s %s] in rms %.4f -> ref rms %.4f, mine %.4f | worst abs %.3e / scale %.3e = %.3e"
          % (L, "dense " if is_dense else "sparse", "local " if is_local else "global",
             x.float().pow(2).mean().sqrt(), y.float().pow(2).mean().sqrt(),
             mine.float().pow(2).mean().sqrt(), d.max(), scale, d.max() / scale))
    del layer, sd
