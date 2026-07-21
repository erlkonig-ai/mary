#!/usr/bin/env python
"""Capture Gemma 4 (E4B) audio-path goldens from HF transformers.

Per reference wav, three stages are saved as .bin (u32 ndim, u32 dims..., f32 LE data):
  {name}.features.bin  — Gemma4AudioFeatureExtractor input_features [T, 128]
  {name}.tower.bin     — Gemma4AudioModel output [T/4, hidden]
  {name}.embed.bin     — Gemma4MultimodalEmbedder output [T/4, text_hidden]

Wavs are pre-trimmed to a multiple of 128 samples so no padded frames exist
(the mask must be all-True; asserted).
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel,
    Gemma4MultimodalEmbedder,
)

SNAP = (os.path.expanduser("~") + "/.cache/huggingface/hub/models--google--gemma-4-E4B-it/"
        "snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c")
WAV_DIR = "/tmp/gemma_audio_work/wavs"
OUT_DIR = "/tmp/gemma_audio_work/goldens"


def save_bin(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(np.uint32(arr.ndim).tobytes())
        f.write(np.array(arr.shape, dtype=np.uint32).tobytes())
        f.write(arr.tobytes())


def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    cfg = AutoConfig.from_pretrained(SNAP)
    audio_cfg = cfg.audio_config
    fe = Gemma4AudioFeatureExtractor()

    print("building tower (f32, cpu)...", flush=True)
    tower = Gemma4AudioModel(audio_cfg).eval().float()
    embedder = Gemma4MultimodalEmbedder(audio_cfg, cfg.text_config).eval().float()

    st_path = os.path.join(SNAP, "model.safetensors")
    tower_sd, emb_sd = {}, {}
    with safe_open(st_path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.audio_tower."):
                tower_sd[k[len("model.audio_tower."):]] = f.get_tensor(k).float()
            elif k.startswith("model.embed_audio."):
                emb_sd[k[len("model.embed_audio."):]] = f.get_tensor(k).float()
    m, u = tower.load_state_dict(tower_sd, strict=False)
    print(f"tower: {len(tower_sd)} loaded, missing={m}, unexpected={u}")
    assert not m and not u, "tower state dict mismatch"
    m, u = embedder.load_state_dict(emb_sd, strict=False)
    print(f"embedder: {len(emb_sd)} loaded, missing={m}, unexpected={u}")
    # embedding_pre_projection_norm is with_scale=False (no params) → both empty
    assert not u, "embedder unexpected keys"

    os.makedirs(OUT_DIR, exist_ok=True)
    for wav in sorted(glob.glob(os.path.join(WAV_DIR, "ref_*.wav"))):
        name = os.path.splitext(os.path.basename(wav))[0]
        x, sr = sf.read(wav, dtype="float32")
        assert sr == 16000 and x.ndim == 1
        batch = fe([x], sampling_rate=16000)
        feats = np.asarray(batch["input_features"], dtype=np.float32)  # [1, T, 128]
        mask = np.asarray(batch["input_features_mask"])                # [1, T]
        assert mask.all(), f"{name}: padded frames present ({mask.sum()}/{mask.size})"

        ft = torch.from_numpy(feats)
        mt = torch.from_numpy(mask).bool()
        out = tower(ft, mt)
        tower_out, tower_mask = out.last_hidden_state, out.attention_mask
        emb_out = embedder(tower_out)
        tower_np = tower_out[0].numpy()
        emb_np = emb_out[0].numpy()
        print(f"{name}: wave {len(x)} → features {feats.shape[1]}×{feats.shape[2]} "
              f"→ tower {tower_np.shape} (mask valid {int(tower_mask.sum())}) "
              f"→ embed {emb_np.shape}")

        save_bin(os.path.join(OUT_DIR, f"{name}.features.bin"), feats[0])
        save_bin(os.path.join(OUT_DIR, f"{name}.tower.bin"), tower_np)
        save_bin(os.path.join(OUT_DIR, f"{name}.embed.bin"), emb_np)

    print("goldens written to", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
