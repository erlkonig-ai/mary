#!/usr/bin/env python3
"""Reference probe: run the official F5-TTS DiT on a fixed input, dump
intermediate activations as .npy for numerical-parity comparison with the
Burn port. CPU/float32 to minimise platform precision noise.

    python3 scripts/probe_f5.py [path-to-model_1250000.safetensors]
"""

import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from f5_tts.model.backbones.dit import DiT

DEFAULT_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/"
    "84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_v1_Base/model_1250000.safetensors"
)
CKPT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probes", "python")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
torch.set_grad_enabled(False)

# ── load weights (strip the ema_model.transformer. prefix into a bare DiT) ──
sd = load_file(CKPT)
pref = "ema_model.transformer."
tsd = {k[len(pref):]: v.float() for k, v in sd.items() if k.startswith(pref)}
text_num_embeds = tsd["text_embed.text_embed.weight"].shape[0] - 1  # table has +1 filler
print(f"text_num_embeds = {text_num_embeds}")

model = DiT(
    dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=100,
    text_num_embeds=text_num_embeds, text_dim=512, conv_layers=4,
    qk_norm=None, text_mask_padding=True, pe_attn_head=None,
)
missing, unexpected = model.load_state_dict(tsd, strict=False)
# non-persistent buffers (rotary inv_freq, text freqs_cis) are recomputed → expected missing
print(f"missing (recomputed buffers ok): {missing}")
print(f"unexpected: {unexpected}")
model.eval()

# ── fixed input ──
B, T, M = 1, 64, 100
x = torch.randn(B, T, M)
cond = torch.randn(B, T, M)
text = torch.randint(0, text_num_embeds, (B, T))  # raw ids; model does +1 internally
time = torch.tensor([0.5])

for name, arr in [("x", x), ("cond", cond), ("text", text.float()), ("time", time)]:
    np.save(os.path.join(OUT, name + ".npy"), arr.numpy().astype(np.float32))

# ── hooks at the same points the Burn forward_probed taps ──
probes = {}

def hook(name):
    def f(_m, _inp, out):
        o = out[0] if isinstance(out, tuple) else out
        probes[name] = o.detach().numpy().astype(np.float32)
    return f

model.input_embed.register_forward_hook(hook("input_embed"))
model.text_embed.register_forward_hook(hook("text_embed"))
model.time_embed.register_forward_hook(hook("time_embed"))
model.transformer_blocks[0].register_forward_hook(hook("block0"))
model.transformer_blocks[21].register_forward_hook(hook("block21"))
model.norm_out.register_forward_hook(hook("norm_out"))
model.proj_out.register_forward_hook(hook("output"))

out = model(x=x, cond=cond, text=text, time=time, drop_audio_cond=False, drop_text=False, cache=False)

for k, v in probes.items():
    np.save(os.path.join(OUT, k + ".npy"), v)
    print(f"{k}: {tuple(v.shape)}")
print(f"output velocity mean {out.mean().item():.4f} std {out.std().item():.4f}")
print(f"✓ reference probes written to {OUT}")
