#!/usr/bin/env python3
"""Reference probe for the Vocos vocoder: export its weights as safetensors for
the Burn port, run it on a fixed mel, and dump intermediates for parity.

    python3 scripts/probe_vocos.py
"""
import os

import numpy as np
import torch
from safetensors.torch import save_file
from vocos import Vocos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "weights")
OUT = os.path.join(ROOT, "probes", "vocos")
os.makedirs(W, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
torch.set_grad_enabled(False)

v = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
save_file({k: t.clone().contiguous() for k, t in v.state_dict().items()}, os.path.join(W, "vocos.safetensors"))
print("exported weights → weights/vocos.safetensors")

T = 64
mel = torch.randn(1, 100, T)
np.save(os.path.join(OUT, "mel.npy"), mel.numpy().astype(np.float32))

probes = {}

def hook(name):
    def f(_m, _i, o):
        probes[name] = (o[0] if isinstance(o, tuple) else o).detach().numpy().astype(np.float32)
    return f

v.backbone.embed.register_forward_hook(hook("embed"))
v.backbone.final_layer_norm.register_forward_hook(hook("backbone"))
v.head.out.register_forward_hook(hook("head_out"))

audio = v.decode(mel)  # backbone → head (ISTFT)
probes["audio"] = audio.detach().numpy().astype(np.float32)

for k, val in probes.items():
    np.save(os.path.join(OUT, k + ".npy"), val)
    print(f"{k}: {tuple(val.shape)}")
print(f"audio len {audio.shape[-1]} = (T-1)*hop = {(T-1)*256}  range [{audio.min():.3f},{audio.max():.3f}]")
