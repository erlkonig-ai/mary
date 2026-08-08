#!/usr/bin/env python3
"""Capture the rest of the Inkling text stack: embedding, head, and a 2-layer run.

Three pieces the per-layer work did not cover:

  * the input side -- `inputs_embeds = embed_norm(embed(ids))`, so the norm is
    applied to the embedding BEFORE any layer sees it;
  * the output side -- `hidden / logits_mup_width_multiplier`, then the head,
    then a truncation from `vocab_size` 201024 to `unpadded_vocab_size` 200058.
    The division is easy to get backwards and the truncation is easy to miss;
    both are checked against values, not against my reading;
  * composition -- two layers in sequence, which is the first check that one
    layer's output feeds the next correctly rather than each being right alone.

The stack is layers 0 and 1, both dense. Going deeper is bounded by the
REFERENCE, not by mary: layer 2's experts alone are 26 GB at f32 and torch
holds the whole stack, while mary pages experts one slab at a time.

  usage: capture_inkling_stack.py <checkpoint dir> <out dir>
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

from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingDecoderLayer, InklingRMSNorm

torch.manual_seed(20260808)
torch.set_default_dtype(torch.float32)

raw = json.load(open(CKPT + "/config.json"))["text_config"]
raw.setdefault("moe_intermediate_size", raw["intermediate_size"])
cfg = InklingTextConfig(**raw)
cfg._attn_implementation = "eager"
H = cfg.hidden_size
T = 8

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


manifest = {
    "tokens": T, "hidden": H,
    "vocab_size": cfg.vocab_size,
    "unpadded_vocab_size": cfg.unpadded_vocab_size,
    "logits_mup_width_multiplier": cfg.logits_mup_width_multiplier,
    "rms_norm_eps": cfg.rms_norm_eps,
    "dense_intermediate": cfg.intermediate_size,
    "sliding_window": cfg.sliding_window_size,
    "kernel": cfg.conv_kernel_size,
    "heads": cfg.swa_num_attention_heads,
    "kv_heads": cfg.swa_num_key_value_heads,
    "head_dim": cfg.swa_head_dim,
    "d_rel": cfg.d_rel,
    "rel_extent": cfg.sliding_window_size,
    "log_scaling_n_floor": cfg.log_scaling_n_floor,
    "log_scaling_alpha": cfg.log_scaling_alpha,
}

# ------------------------------------------------------------- embedding ---
ids = torch.tensor([1, 2, 100, 12345, 199999, 0, 77, 200057], dtype=torch.long)
assert ids.numel() == T
assert int(ids.max()) < cfg.unpadded_vocab_size, "token id past the unpadded vocab"
embed_w = get("model.llm.embed.weight")
embed_norm_w = get("model.llm.embed_norm.weight")
norm = InklingRMSNorm(H, eps=cfg.rms_norm_eps)
with torch.no_grad():
    norm.weight.copy_(embed_norm_w)
    raw_embed = torch.nn.functional.embedding(ids, embed_w)
    inputs_embeds = norm(raw_embed)
w("stk_ids_f32.bin", ids.float())
w("stk_raw_embed.bin", raw_embed)
w("stk_inputs_embeds.bin", inputs_embeds)
open(os.path.join(OUT, "stk_ids.bin"), "wb").write(
    np.ascontiguousarray(ids.numpy().astype("<i8")).tobytes())
print("embedding: ids %s -> %s  |raw| max %.6g  |normed| max %.6g"
      % (ids.tolist(), tuple(inputs_embeds.shape), raw_embed.abs().max(), inputs_embeds.abs().max()))
# The norm must actually change the embedding, or this check is vacuous.
delta = (inputs_embeds - raw_embed).abs().max()
manifest["embed_norm_delta"] = float(delta)
print("          embed_norm moves it by %.6g" % delta)

# ------------------------------------------------------------------ head ---
final_norm_w = get("model.llm.norm.weight")
unembed_w = get("model.llm.unembed.weight")
hid = torch.randn(T, H) * 0.5
fnorm = InklingRMSNorm(H, eps=cfg.rms_norm_eps)
with torch.no_grad():
    fnorm.weight.copy_(final_norm_w)
    normed = fnorm(hid)
    scaled = normed / cfg.logits_mup_width_multiplier
    logits_full = torch.nn.functional.linear(scaled, unembed_w)
    logits = logits_full[..., : cfg.unpadded_vocab_size]
w("stk_head_in.bin", hid)
w("stk_logits.bin", logits)
print("head: hidden %s -> logits %s (from %d, truncated to %d)"
      % (tuple(hid.shape), tuple(logits.shape), logits_full.shape[-1], cfg.unpadded_vocab_size))
print("      mup divisor %.6g ; |logits| max %.6g" % (cfg.logits_mup_width_multiplier, logits.abs().max()))
# Was anything actually dropped? If vocab_size == unpadded, the truncation is
# untested and a gate that skips it would pass.
manifest["logits_dropped"] = int(logits_full.shape[-1] - cfg.unpadded_vocab_size)
print("      truncation dropped %d columns" % manifest["logits_dropped"])
# What the logits would be WITHOUT the mup divide, so the gate can require the
# two to differ and therefore that the divide is under test.
w("stk_logits_nomup.bin", torch.nn.functional.linear(normed, unembed_w)[..., : cfg.unpadded_vocab_size])

# ---------------------------------------------------------- 2-layer stack ---
def load_dense(idx):
    p = f"model.llm.layers.{idx}."
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
    layer = InklingDecoderLayer(cfg, idx)
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return layer


assert cfg.mlp_layer_types[0] == "dense" and cfg.mlp_layer_types[1] == "dense"
assert cfg.layer_types[0] == "hybrid_sliding" and cfg.layer_types[1] == "hybrid_sliding"

x = inputs_embeds.unsqueeze(0)          # feed the REAL embedding, not noise
mask = create_sliding_window_causal_mask(
    config=cfg, inputs_embeds=x, attention_mask=None,
    past_key_values=None, position_ids=torch.arange(T).unsqueeze(0))

h = x
per_layer = []
for idx in (0, 1):
    layer = load_dense(idx)
    with torch.no_grad():
        h = layer(h, attention_mask=mask, conv_mask=None, past_key_values=None)
    per_layer.append(h.clone())
    print("stack: after layer %d, |h| max %.6g" % (idx, h.abs().max()))
    del layer

w("stk_after0.bin", per_layer[0])
w("stk_after1.bin", per_layer[1])
# The two must differ, or "composed two layers" would be indistinguishable from
# "ran one layer twice and compared the wrong one".
manifest["stack_delta"] = float((per_layer[1] - per_layer[0]).abs().max())
print("       layer 1 moved the stream by %.6g" % manifest["stack_delta"])

json.dump(manifest, open(os.path.join(OUT, "stk_manifest.json"), "w"), indent=1)
print("wrote oracle to", OUT)
