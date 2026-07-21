#!/usr/bin/env python3
"""De-risk reference dump for the IMAGE embedding path of
nomic-embed-multimodal-7b (BiQwen2_5). Dumps the intermediate anchors that let
the Rust port verify get_rope_index, the multimodal splice, and section-wise
M-RoPE INDEPENDENTLY of the final embedding:

  image_input_ids.npy      [1, S]      the 15-token image prompt ids
  image_position_ids.npy   [3, 1, S]   M-RoPE 3D position ids (t/h/w)
  image_inputs_embeds.npy  [1, S, H]   text embeds with vision tokens spliced in
  image_vision_tokens.npy  [n, H]      the merged vision tokens that got spliced
  image_last_hidden.npy    [1, S, H]   text-backbone last_hidden_state (pre-pool)
  image_emb.npy            [1, H]      final L2 dense embedding (re-confirm)

Hooks the language-model submodule with a forward-pre-hook so we capture the
EXACT inputs_embeds + position_ids HF builds internally, independent of any
private API. Run: python3 scripts/nomic_mm7b_image_dump.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

MODEL_ID = "nomic-ai/nomic-embed-multimodal-7b"
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "nomic_mm7b"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, t):
    np.save(OUT / f"{name}.npy", t.detach().cpu().float().numpy())


torch.manual_seed(0)
dtype = torch.float32

print("loading processor + model (base Qwen2.5-VL-7B + LoRA, fp32 CPU)...")
proc = BiQwen2_5_Processor.from_pretrained(MODEL_ID)
model = BiQwen2_5.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="cpu").eval()

img = Image.new("RGB", (56, 56), color=(123, 200, 90))
bi = proc.process_images([img])
print("image_input_ids:", bi["input_ids"][0].tolist())
print("image_grid_thw :", bi["image_grid_thw"].tolist())
# float32 (the Rust npy loader is f32-only; ids are small & exactly representable)
np.save(OUT / "image_input_ids.npy", bi["input_ids"].cpu().numpy().astype(np.float32))

# --- locate the text decoder submodule to hook ------------------------------
# BiQwen2_5 wraps a Qwen2_5_VL*Model; the text decoder is the module that
# receives inputs_embeds + position_ids. Probe a few known attribute paths.
def find_language_model(m):
    cands = [
        "model.language_model",
        "model.model.language_model",
        "model.model",
        "language_model",
    ]
    for path in cands:
        obj = m
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, "forward"):
            # heuristic: the text decoder has `layers` (decoder stack)
            if hasattr(obj, "layers"):
                print(f"hooking language model at: {path}  ({type(obj).__name__})")
                return obj
    raise RuntimeError("could not locate language-model submodule")


lm = find_language_model(model)

captured = {}


def pre_hook(module, args, kwargs):
    # capture inputs_embeds + position_ids however they arrive (args or kwargs)
    ie = kwargs.get("inputs_embeds")
    pid = kwargs.get("position_ids")
    if ie is None and len(args) >= 2:
        ie = args[1]
    captured["inputs_embeds"] = ie
    captured["position_ids"] = pid


h = lm.register_forward_pre_hook(pre_hook, with_kwargs=True)


def post_hook(module, args, output):
    lhs = getattr(output, "last_hidden_state", None)
    if lhs is None and isinstance(output, (tuple, list)):
        lhs = output[0]
    captured["last_hidden"] = lhs


h2 = lm.register_forward_hook(post_hook)

# also capture the visual tower output (the merged image tokens to be spliced)
vis = None
for path in ["model.visual", "model.model.visual", "visual"]:
    obj = model
    ok = True
    for part in path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            ok = False
            break
    if ok:
        vis = obj
        print(f"hooking visual at: {path} ({type(obj).__name__})")
        break

vis_out = {}
if vis is not None:
    vis.register_forward_hook(lambda m, i, o: vis_out.__setitem__("out", o))

def first_tensor(o):
    if o is None:
        return None
    if torch.is_tensor(o):
        return o
    if hasattr(o, "pooler_output") and torch.is_tensor(getattr(o, "pooler_output")):
        return o.pooler_output
    if hasattr(o, "last_hidden_state") and torch.is_tensor(getattr(o, "last_hidden_state")):
        return o.last_hidden_state
    if isinstance(o, (tuple, list)):
        for x in o:
            t = first_tensor(x)
            if t is not None:
                return t
    return None


with torch.no_grad():
    emb = model(**bi)  # BiQwen forward -> pooled + L2 normalized

h.remove()
save("image_emb", emb)
print("image_emb shape:", tuple(emb.shape))

ie = captured["inputs_embeds"]
pid = captured["position_ids"]
print("captured inputs_embeds:", None if ie is None else tuple(ie.shape))
print("captured position_ids :", None if pid is None else tuple(pid.shape))
if ie is not None:
    save("image_inputs_embeds", ie)
if pid is not None:
    save("image_position_ids", pid)
    # human-readable for sanity
    p = pid.detach().cpu().long().numpy()
    print("position_ids (t/h/w) per token:")
    print(p.reshape(p.shape[0], -1))

# --- text last_hidden_state for the image sequence (anchor before pooling) ---
lhs = first_tensor(captured.get("last_hidden"))
if lhs is not None:
    save("image_last_hidden", lhs)
    print("image_last_hidden shape:", tuple(lhs.shape))
h2.remove()

if "out" in vis_out:
    vt = first_tensor(vis_out["out"])
    if vt is not None:
        save("image_vision_tokens", vt)
        print("vision tokens (spliced) shape:", tuple(vt.shape))

print("done. goldens ->", OUT)
