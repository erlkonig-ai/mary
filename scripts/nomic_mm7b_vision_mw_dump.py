#!/usr/bin/env python3
"""Multi-window vision golden for the Qwen2.5-VL tower of
nomic-embed-multimodal-7b. The 56x56 single-window probe (vision_dump.py) leaves
the window-partition / cu_window_seqlens / scatter paths un-pinned: its 2x2
merged grid fits one 4x4 (merged-unit) window. This dumps a 140x140 image whose
10x10 patch grid -> 5x5 merged units SPANS 2 windows per axis (4 windows, with
padding), so the windowed-attention reorder + scatter actually fire.

Dumps to tests/golden/nomic_mm7b/vision_mw/:  pixel_values, grid_thw, merger_out.
Run: python3 scripts/nomic_mm7b_vision_mw_dump.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)

BASE = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*"))[0]
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "nomic_mm7b" / "vision_mw"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, t):
    np.save(OUT / f"{name}.npy", t.detach().cpu().float().numpy())


torch.manual_seed(0)
dtype = torch.float32
cfg = AutoConfig.from_pretrained(BASE)
vcfg = cfg.vision_config

index = json.load(open(Path(BASE) / "model.safetensors.index.json"))["weight_map"]
vis_keys = [k for k in index if k.startswith("visual.")]
by_shard: dict[str, list[str]] = {}
for k in vis_keys:
    by_shard.setdefault(index[k], []).append(k)
state = {}
for shard, keys in by_shard.items():
    with safe_open(str(Path(BASE) / shard), "pt") as s:
        for k in keys:
            state[k] = s.get_tensor(k).float()

vt = Qwen2_5_VisionTransformerPretrainedModel(vcfg).to(dtype).eval()
vt.load_state_dict({k[len("visual."):]: v for k, v in state.items()}, strict=False)

# 140x140 -> 10x10 patches -> 5x5 merged units (> window 4) -> multi-window.
proc = AutoProcessor.from_pretrained(BASE)
img = Image.new("RGB", (140, 140), color=(40, 160, 220))
vp = proc.image_processor(images=[img], return_tensors="pt")
pixel_values = vp["pixel_values"].to(dtype)
grid_thw = vp["image_grid_thw"]
print("pixel_values", tuple(pixel_values.shape), "grid_thw", grid_thw.tolist())
assert grid_thw.tolist() == [[1, 10, 10]], f"unexpected grid {grid_thw.tolist()}"
save("pixel_values", pixel_values)
np.save(OUT / "grid_thw.npy", grid_thw.cpu().numpy().astype(np.float32))

with torch.no_grad():
    out = vt(pixel_values, grid_thw)
save("merger_out", out.pooler_output)
print("merger_out (pooler) shape:", tuple(out.pooler_output.shape))
print("goldens ->", OUT)
