#!/usr/bin/env python3
"""Merge the nomic-embed-multimodal-7b LoRA adapter into the Qwen2.5-VL-7B base
text backbone and emit a single f16 safetensors of JUST the text decoder, keyed
in the `QwenTextModel` naming (`embed_tokens.weight`, `layers.{i}...`,
`norm.weight`).

The adapter is standard PEFT LoRA (r=32, alpha=32 -> scale=1.0) on the seven text
projections (q/k/v/o_proj, gate/up/down_proj) of all 28 layers. PEFT's merge is
exactly  W' = W + (alpha/r) * (lora_B @ lora_A)  for each target Linear; we do it
tensor-by-tensor straight from safetensors (low RAM, no 16GB torch model).

Base text keys are `model.layers.N.*`, `model.embed_tokens.weight`,
`model.norm.weight`; we strip the leading `model.` so the result loads directly
into `mary::models::qwen2_5_vl::layers::QwenTextModel`.

Run: python3 scripts/nomic_mm7b_merge.py /path/to/merged_text_backbone.safetensors
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*"))[0]
ADAPTER = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--nomic-ai--nomic-embed-multimodal-7b/snapshots/*/adapter_model.safetensors"))[0]
ADAPTER_CFG = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--nomic-ai--nomic-embed-multimodal-7b/snapshots/*/adapter_config.json"))[0]

out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/private/tmp/merged_text_backbone.safetensors")
out_path.parent.mkdir(parents=True, exist_ok=True)

cfg = json.load(open(ADAPTER_CFG))
scale = cfg["lora_alpha"] / cfg["r"]
print(f"LoRA scale alpha/r = {cfg['lora_alpha']}/{cfg['r']} = {scale}")

# --- index the base text-backbone tensors across shards ---
index = json.load(open(Path(BASE) / "model.safetensors.index.json"))["weight_map"]
TEXT_PREFIX = "model."  # text backbone; vision is under `visual.`
def is_text(k: str) -> bool:
    return (k.startswith("model.layers.") or k == "model.embed_tokens.weight"
            or k == "model.norm.weight")
text_keys = [k for k in index if is_text(k)]
print(f"base text-backbone tensors: {len(text_keys)}")

# group base keys by shard so we open each shard once
by_shard: dict[str, list[str]] = {}
for k in text_keys:
    by_shard.setdefault(index[k], []).append(k)

# --- load all LoRA A/B for the targeted modules (small: 308MB, fp32) ---
lora: dict[str, dict[str, torch.Tensor]] = {}
with safe_open(ADAPTER, "pt") as s:
    for ak in s.keys():
        # base_model.model.model.layers.N.<mod>.lora_{A,B}.weight  ->  layers.N.<mod>.weight
        if ".lora_A.weight" in ak:
            kind, base_key = "A", ak
        elif ".lora_B.weight" in ak:
            kind, base_key = "B", ak
        else:
            continue
        # strip PEFT wrapper prefix + lora suffix to recover the merged target key
        tgt = base_key.replace("base_model.model.model.", "").replace(f".lora_{kind}.weight", ".weight")
        lora.setdefault(tgt, {})[kind] = s.get_tensor(ak).float()
print(f"LoRA target modules: {len(lora)} (expect 28*7 = 196)")

# --- merge, strip `model.` prefix, cast f16, collect ---
merged: dict[str, torch.Tensor] = {}
n_merged = 0
for shard, keys in by_shard.items():
    with safe_open(str(Path(BASE) / shard), "pt") as s:
        for k in keys:
            w = s.get_tensor(k).float()  # bf16 -> f32 for clean math
            tgt = k[len(TEXT_PREFIX):]   # `model.layers.0...` -> `layers.0...`; `model.norm.weight` -> `norm.weight`
            if tgt in lora:
                A = lora[tgt]["A"]  # [r, in]
                B = lora[tgt]["B"]  # [out, r]
                w = w + scale * (B @ A)
                n_merged += 1
            merged[tgt] = w.to(torch.float16).contiguous()
print(f"merged {n_merged} LoRA deltas into base (expect 196)")
assert n_merged == len(lora), f"merge count {n_merged} != lora targets {len(lora)}"

save_file(merged, str(out_path))
print(f"wrote {len(merged)} f16 tensors -> {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")
