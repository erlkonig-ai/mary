#!/usr/bin/env python3
"""Capture the Inkling vision (HMLP) and audio (dMel) towers on real weights.

Both config classes want `text_hidden_size`, which the checkpoint calls
`decoder_dmodel` and does not map — so without help they fall back to the class
default of 6144, the 66-layer model's width, and the 42-layer release silently
gets full-model dimensions. This is the same gap as `moe_intermediate_size` on
the text side: the transformers defaults are the big model's values.

  usage: capture_inkling_towers.py <checkpoint dir> <out dir>
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

from transformers.models.inkling.configuration_inkling import (
    InklingAudioConfig,
    InklingVisionConfig,
)
from transformers.models.inkling.modeling_inkling import InklingAudioModel, InklingVisionModel

torch.manual_seed(20260808)
torch.set_default_dtype(torch.float32)

cfgj = json.load(open(CKPT + "/config.json"))
vraw = dict(cfgj["vision_config"])
araw = dict(cfgj["audio_config"])
TEXT_H = cfgj["text_config"]["hidden_size"]

# The unmapped field, supplied explicitly. Assert it took, rather than trusting.
vraw["text_hidden_size"] = vraw.pop("decoder_dmodel", TEXT_H)
araw["text_hidden_size"] = araw.pop("decoder_dmodel", TEXT_H)
vraw.pop("vision_encoder_type", None)
vraw.pop("use_vision_norm", None)
vraw["num_channels"] = vraw.pop("n_channels", 3)
for k in ("bias", "dmel_min_value", "dmel_max_value", "use_audio_norm", "audio_mode"):
    araw.pop(k, None)

vcfg = InklingVisionConfig(**vraw)
acfg = InklingAudioConfig(**araw)
assert vcfg.text_hidden_size == TEXT_H, (vcfg.text_hidden_size, TEXT_H)
assert acfg.text_hidden_size == TEXT_H, (acfg.text_hidden_size, TEXT_H)
print("vision: patch %d, temporal %d, layers %d, channels %d -> text %d"
      % (vcfg.patch_size, vcfg.temporal_patch_size, vcfg.num_hidden_layers,
         vcfg.num_channels, vcfg.text_hidden_size))
print("audio : %d bins x %d levels -> text %d"
      % (acfg.num_codebooks, acfg.codebook_size, acfg.text_hidden_size))

weight_map = json.load(open(CKPT + "/model.safetensors.index.json"))["weight_map"]


def get(name):
    with safe_open(CKPT + "/" + weight_map[name], framework="pt") as f:
        return f.get_tensor(name).float()


def w(name, t):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
    open(os.path.join(OUT, name), "wb").write(a.tobytes())


manifest = {
    "patch_size": vcfg.patch_size,
    "temporal_patch_size": vcfg.temporal_patch_size,
    "n_layers": vcfg.num_hidden_layers,
    "n_channels": vcfg.num_channels,
    "text_hidden": TEXT_H,
    "rms_norm_eps": vcfg.rms_norm_eps,
    "n_mel_bins": acfg.num_codebooks,
    "mel_levels": acfg.codebook_size,
}

# ------------------------------------------------------------------ vision ---
vm = InklingVisionModel(vcfg)
scales = vm.scales.tolist()
print("planned scales (t,h,w,c):", scales)
manifest["scales"] = scales
sd = {}
for i, layer in enumerate(vm.encoder_layers):
    sd[f"encoder_layers.{i}.projection.weight"] = get(f"model.visual.layers.linear_{i}.weight")
    if layer.add_norm:
        sd[f"encoder_layers.{i}.layer_norm.weight"] = get(f"model.visual.layers.norm_{i}.weight")
sd["final_norm.weight"] = get("model.visual.final_norm.weight")
missing, unexpected = vm.load_state_dict(sd, strict=False)
print("vision state_dict: %d mapped, %d missing, %d unexpected" % (len(sd), len(missing), len(unexpected)))
assert not missing and not unexpected, (sorted(missing)[:6], sorted(unexpected)[:6])

P = 3
px = torch.randn(P, vcfg.temporal_patch_size, vcfg.patch_size, vcfg.patch_size, vcfg.num_channels)
with torch.no_grad():
    vout = vm(px).last_hidden_state
w("tow_px.bin", px)
w("tow_vision_y.bin", vout)
print("vision: %s -> %s  |y| max %.6g" % (tuple(px.shape), tuple(vout.shape), vout.abs().max()))
manifest["patches"] = P
manifest["vision_out"] = list(vout.shape)
stage_info = []
for i, layer in enumerate(vm.encoder_layers):
    stage_info.append({
        "t_fold": int(layer.t_fold), "hw_fold": int(layer.hw_fold),
        "in": int(layer.projection.in_features), "out": int(layer.projection.out_features),
        "add_norm": bool(layer.add_norm),
    })
    print("  stage %d: fold t=%d hw=%d, %d -> %d, norm=%s"
          % (i, layer.t_fold, layer.hw_fold, layer.projection.in_features,
             layer.projection.out_features, layer.add_norm))
manifest["stages"] = stage_info
# Non-vacuity: at least one stage must fold spatially and one temporally, or
# fold_timespace_to_depth is not really under test.
manifest["stages_folding_hw"] = sum(1 for s in stage_info if s["hw_fold"] > 1)
manifest["stages_folding_t"] = sum(1 for s in stage_info if s["t_fold"] > 1)

# ------------------------------------------------------------------- audio ---
am = InklingAudioModel(acfg)
asd = {
    "embed_audio_tokens.embed_audio_tokens.weight": get("model.audio.encoder.weight"),
    "norm.weight": get("model.audio.final_norm.weight"),
}
missing, unexpected = am.load_state_dict(asd, strict=False)
print("audio state_dict: %d mapped, %d missing, %d unexpected" % (len(asd), len(missing), len(unexpected)))
assert not missing and not unexpected, (sorted(missing)[:6], sorted(unexpected)[:6])

F = 5
ids = torch.randint(0, acfg.codebook_size, (F, acfg.num_codebooks), dtype=torch.long)
with torch.no_grad():
    aout = am(ids).last_hidden_state
open(os.path.join(OUT, "tow_audio_ids.bin"), "wb").write(
    np.ascontiguousarray(ids.numpy().astype("<i8")).tobytes())
w("tow_audio_y.bin", aout)
print("audio: ids %s -> %s  |y| max %.6g" % (tuple(ids.shape), tuple(aout.shape), aout.abs().max()))
manifest["frames"] = F
# Non-vacuity: the lookup must hit many distinct rows, not one.
manifest["distinct_levels_used"] = int(len(torch.unique(ids)))
print("       distinct mel levels used: %d of %d" % (manifest["distinct_levels_used"], acfg.codebook_size))

sc = np.array(scales, dtype="<i8")
open(os.path.join(OUT, "tow_scales.bin"), "wb").write(np.ascontiguousarray(sc).tobytes())
st = np.array([[s["t_fold"], s["hw_fold"], s["in"], s["out"], int(s["add_norm"])]
               for s in stage_info], dtype="<i8")
open(os.path.join(OUT, "tow_stages.bin"), "wb").write(np.ascontiguousarray(st).tobytes())
print("dumped %d scales x 4 and %d stages x 5" % (sc.shape[0], st.shape[0]))

json.dump(manifest, open(os.path.join(OUT, "tow_manifest.json"), "w"), indent=1)
print("wrote oracle to", OUT)
