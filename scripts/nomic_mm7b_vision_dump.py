#!/usr/bin/env python3
"""Reference goldens for the Qwen2.5-VL vision tower (the vision encoder of
nomic-embed-multimodal-7b — the adapter has NO vision LoRA, so the base
`visual.*` weights ARE final).

Instantiates ONLY the vision transformer (~675M, not the 16GB full model),
loads `visual.*` from the base safetensors, runs the 56x56 probe image, and
dumps float32 intermediates for the Rust parity harness:
  pixel_values, grid_thw, patch_embed_out, rot_cos/rot_sin, block0_out,
  vision_last_hidden (pre-merger), merger_out (pooler_output, [n_merged, 3584]).

Also writes a vision-only f16 safetensors (keys stripped of the `visual.`
prefix) for the pile round-trip.

Run: python3 scripts/nomic_mm7b_vision_dump.py [<vision_safetensors_out>]
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)

BASE = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*"))[0]
MODEL_ID = "nomic-ai/nomic-embed-multimodal-7b"
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "nomic_mm7b" / "vision"
OUT.mkdir(parents=True, exist_ok=True)
vis_st_out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/private/tmp/vision_merged/vision_tower.safetensors")
vis_st_out.parent.mkdir(parents=True, exist_ok=True)

def save(name, t):
    np.save(OUT / f"{name}.npy", t.detach().cpu().float().numpy())

torch.manual_seed(0)
dtype = torch.float32

cfg = AutoConfig.from_pretrained(BASE)
vcfg = cfg.vision_config
print("vision_config:", vcfg)

# --- load visual.* weights from the base safetensors ---
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
print(f"loaded {len(state)} visual.* tensors")

vt = Qwen2_5_VisionTransformerPretrainedModel(vcfg).to(dtype).eval()
missing, unexpected = vt.load_state_dict({k[len("visual."):]: v for k, v in state.items()}, strict=False)
print("missing:", missing[:4], "... unexpected:", unexpected[:4])

# --- build the probe image's pixel_values + grid_thw ---
proc = AutoProcessor.from_pretrained(BASE)
img = Image.new("RGB", (56, 56), color=(123, 200, 90))
vp = proc.image_processor(images=[img], return_tensors="pt")
pixel_values = vp["pixel_values"].to(dtype)
grid_thw = vp["image_grid_thw"]
print("pixel_values", pixel_values.shape, "grid_thw", grid_thw.tolist())
save("pixel_values", pixel_values)
np.save(OUT / "grid_thw.npy", grid_thw.cpu().numpy().astype(np.int64))

# --- hooks for intermediates ---
acts = {}
vt.patch_embed.register_forward_hook(lambda m, i, o: acts.__setitem__("patch_embed_out", o))
vt.blocks[0].register_forward_hook(lambda m, i, o: acts.__setitem__("block0_out", o))

with torch.no_grad():
    # replicate forward to capture rot cos/sin too
    out = vt(pixel_values, grid_thw)
save("patch_embed_out", acts["patch_embed_out"])
save("block0_out", acts["block0_out"])
save("vision_last_hidden", out.last_hidden_state)
save("merger_out", out.pooler_output)
print("merger_out (pooler) shape:", tuple(out.pooler_output.shape))

# rotary cos/sin (recompute exactly as forward does)
with torch.no_grad():
    rot = vt.rot_pos_emb(grid_thw)
    emb = torch.cat((rot, rot), dim=-1)
save("rot_cos", emb.cos())
save("rot_sin", emb.sin())

# --- write vision-only f16 safetensors (strip `visual.` prefix) ---
f16 = {k[len("visual."):]: v.to(torch.float16).contiguous() for k, v in state.items()}
save_file(f16, str(vis_st_out))
print(f"wrote {len(f16)} f16 vision tensors -> {vis_st_out} ({vis_st_out.stat().st_size/1e9:.2f} GB)")
print("goldens ->", OUT)
