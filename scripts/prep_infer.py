#!/usr/bin/env python3
"""End-to-end prep + reference: tokenize ref+gen text, extract the reference mel,
replicate F5's inference (duration estimate, 32-step CFM, slice gen, vocode) with
a FIXED y0, and save every intermediate so the Burn `speak` binary can reproduce
the waveform and play it. Also writes the reference wav for parity.

    python3 scripts/prep_infer.py [path-to-model.safetensors]
"""
import math
import os
import sys

import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.modules import get_vocos_mel_spectrogram
from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx, get_tokenizer
from vocos import Vocos

DEFAULT_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/"
    "84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_v1_Base/model_1250000.safetensors"
)
CKPT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probes", "infer")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
torch.set_grad_enabled(False)

import f5_tts.model.utils as _u
F5DIR = os.path.dirname(os.path.dirname(_u.__file__))
REF_WAV = os.path.join(F5DIR, "infer/examples/basic/basic_ref_en.wav")
REF_TEXT = "Some call me nature, others call me mother nature."
GEN_TEXT = "I don't really care what you call me. I am the voice of the body now."
VOCAB = os.path.join(F5DIR, "infer/examples/vocab.txt")
HOP, SR, STEPS, CFG, SWAY = 256, 24000, 32, 2.0, -1.0

# ── model ──
sd = load_file(CKPT)
pref = "ema_model.transformer."
tsd = {k[len(pref):]: v.float() for k, v in sd.items() if k.startswith(pref)}
tne = tsd["text_embed.text_embed.weight"].shape[0] - 1
dit = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=100,
          text_num_embeds=tne, text_dim=512, conv_layers=4, qk_norm=None,
          text_mask_padding=True, pe_attn_head=None)
dit.load_state_dict(tsd, strict=False)
dit.eval()
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
vocab_map, _ = None, None
with open(VOCAB, "r", encoding="utf-8") as f:
    vocab_map = {c[:-1]: i for i, c in enumerate(f)}

# ── ref mel ──
audio, sr = torchaudio.load(REF_WAV)
if sr != SR:
    audio = torchaudio.transforms.Resample(sr, SR)(audio)
if audio.shape[0] > 1:
    audio = audio.mean(0, keepdim=True)
ref_mel = get_vocos_mel_spectrogram(audio, n_fft=1024, n_mel_channels=100, target_sample_rate=SR,
                                    hop_length=HOP, win_length=1024)  # [1,100,T]
ref_mel = ref_mel.permute(0, 2, 1)  # [1,T,100]
ref_len = ref_mel.shape[1]
np.save(os.path.join(OUT, "ref_audio.npy"), audio.numpy().astype(np.float32))  # [1, n_samples]

# ── tokenize + duration ──
text_list = convert_char_to_pinyin([REF_TEXT + " " + GEN_TEXT])
text_ids = list_str_to_idx(text_list, vocab_map)  # [1, n_chars]
rb, gb = len(REF_TEXT.encode()), len(GEN_TEXT.encode())
speed = 1.0
duration = ref_len + int(ref_len / rb * gb / speed)
print(f"ref_len {ref_len}  duration {duration}  n_chars {text_ids.shape[1]}")

# ── cond + fixed noise ──
cond = torch.nn.functional.pad(ref_mel, (0, 0, 0, duration - ref_len), value=0.0)  # [1,dur,100]
y0 = torch.randn(1, duration, 100)

t = torch.linspace(0, 1, STEPS + 1)
t = t + SWAY * (torch.cos(math.pi / 2 * t) - 1 + t)

def velocity(x, tt):
    pred = dit(x=x, cond=cond, text=text_ids, time=tt, drop_audio_cond=False, drop_text=False, cache=False)
    null = dit(x=x, cond=cond, text=text_ids, time=tt, drop_audio_cond=True, drop_text=True, cache=False)
    return pred + (pred - null) * CFG

# single-forward parity check on the real (T=duration, padded-text) input
v0 = dit(x=y0, cond=cond, text=text_ids, time=torch.zeros(1), drop_audio_cond=False, drop_text=False, cache=False)
np.save(os.path.join(OUT, "v0_cond.npy"), v0.numpy().astype(np.float32))

x = y0.clone()
for i in range(STEPS):
    x = x + (t[i + 1] - t[i]) * velocity(x, t[i].repeat(1))
np.save(os.path.join(OUT, "sampled_mel.npy"), x.numpy().astype(np.float32))
gen_mel = x[:, ref_len:, :].permute(0, 2, 1)  # [1,100,gen_T]
wave = vocos.decode(gen_mel).squeeze().cpu().numpy()

# ── save ──
np.save(os.path.join(OUT, "ref_mel.npy"), ref_mel.numpy().astype(np.float32))
np.save(os.path.join(OUT, "text_ids.npy"), text_ids.numpy().astype(np.float32))
np.save(os.path.join(OUT, "y0.npy"), y0.numpy().astype(np.float32))
np.save(os.path.join(OUT, "ref_wave.npy"), wave.astype(np.float32))
with open(os.path.join(OUT, "meta.txt"), "w") as f:
    f.write(f"{ref_len} {duration}\n")
torchaudio.save(os.path.join(OUT, "ref_wave.wav"), torch.tensor(wave).unsqueeze(0), SR)
print(f"wave {wave.shape} [{wave.min():.3f},{wave.max():.3f}]  → {OUT}/ref_wave.wav")
