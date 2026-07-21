#!/usr/bin/env python3
"""Reference probe for the CFM sampler: replicate F5's CFM.sample integration
(linspace+sway grid, Euler, CFG cond+(cond-uncond)*cfg) directly over the
validated DiT, with a FIXED initial noise y0, and dump the final mel.

    python3 scripts/probe_cfm.py [path-to-model.safetensors]
"""
import math
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
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probes", "cfm")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
torch.set_grad_enabled(False)

sd = load_file(CKPT)
pref = "ema_model.transformer."
tsd = {k[len(pref):]: v.float() for k, v in sd.items() if k.startswith(pref)}
tne = tsd["text_embed.text_embed.weight"].shape[0] - 1
dit = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=100,
          text_num_embeds=tne, text_dim=512, conv_layers=4, qk_norm=None,
          text_mask_padding=True, pe_attn_head=None)
dit.load_state_dict(tsd, strict=False)
dit.eval()

B, T, M = 1, 48, 100
STEPS, CFG, SWAY = 8, 2.0, -1.0
cond = torch.randn(B, T, M)
text = torch.randint(0, tne, (B, T))
y0 = torch.randn(B, T, M)
for n, a in [("cond", cond), ("text", text.float()), ("y0", y0)]:
    np.save(os.path.join(OUT, n + ".npy"), a.numpy().astype(np.float32))

# time grid: linspace then sway
t = torch.linspace(0, 1, STEPS + 1)
t = t + SWAY * (torch.cos(math.pi / 2 * t) - 1 + t)

def velocity(x, tt):
    pred = dit(x=x, cond=cond, text=text, time=tt, drop_audio_cond=False, drop_text=False, cache=False)
    null = dit(x=x, cond=cond, text=text, time=tt, drop_audio_cond=True, drop_text=True, cache=False)
    return pred + (pred - null) * CFG

x = y0.clone()
for i in range(STEPS):
    tt = t[i].repeat(B)
    x = x + (t[i + 1] - t[i]) * velocity(x, tt)

np.save(os.path.join(OUT, "mel.npy"), x.numpy().astype(np.float32))
print(f"CFM reference mel {tuple(x.shape)} mean {x.mean():.4f} std {x.std():.4f}")
print(f"✓ → {OUT}")
