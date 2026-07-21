# Build a packaged PersonaPlex voice-prompt .pt from a WAV — the upstream
# flow, driven directly (Phase 5).
#
# This is NOT a converter we invented: it calls the exact code path the NVIDIA
# reference ships for turning a voice WAV into the packaged
# {embeddings, cache} .pt that `load_voice_prompt_embeddings` replays
# (`moshi/models/lm.py`: `LMGen.load_voice_prompt` + the
# `save_voice_prompt_embeddings=True` save branch of `_step_voice_prompt_core`;
# `moshi/offline.py run()` exposes the same parameter and accepts non-`.pt`
# voice prompts). The flow: load WAV -> normalize to -24 LUFS (pyloudnorm, an
# UNDECLARED upstream dep — `pip install pyloudnorm` into the venv) -> Mimi
# streaming encode -> one LM step per frame (agent stream = voice codes, user
# stream = SINE_TOKENS, text = PAD 3) with the input embeddings recorded ->
# torch.save({"embeddings": [N,1,1,4096], "cache": [1,17,4]}, <wav-stem>.pt).
#
# CPU float32 (same numeric family as the parity goldens; the stock voices
# were saved from a CUDA bf16 run, hence their bf16 embeddings + cuda:0
# storages). ~3 s/step on CPU: a 10 s WAV = 130 frames ≈ 7 min + ~2 min load.
#
# Usage (from the moshi venv; the .pt lands next to the INPUT wav, so copy
# the wav to a scratch dir first):
#   /tmp/personaplex_scratch/moshi_venv/bin/python golden/build_voice_prompt.py \
#     /tmp/mary-personaplex/ref_voice.wav
import os
import sys
import time

import torch

CKPT = "/tmp/personaplex_scratch/ckpt"


def main():
    wav = sys.argv[1]
    assert not wav.endswith(".pt"), "input must be a WAV (the .pt is the output)"
    from moshi.models import loaders, LMGen

    t0 = time.time()
    print("loading mimi (cpu)...")
    mimi = loaders.get_mimi(os.path.join(CKPT, loaders.MIMI_NAME), "cpu")
    print(f"mimi loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("loading moshi LM (cpu, float32)...")
    lm = loaders.get_moshi_lm(
        os.path.join(CKPT, loaders.MOSHI_NAME), device="cpu", dtype=torch.float32
    )
    lm.eval()
    print(f"moshi LM loaded in {time.time()-t0:.1f}s")

    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * 12.5),
        sample_rate=mimi.sample_rate,
        device="cpu",
        frame_rate=12.5,
        use_sampling=False,  # nothing sampled lands in the saved cache anyway
        save_voice_prompt_embeddings=True,
    )
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    lm_gen.load_voice_prompt(wav)  # load + -24 LUFS normalize (upstream)
    n_frames = lm_gen.voice_prompt_audio.shape[-1] / lm_gen._frame_size
    print(f"voice wav: {lm_gen.voice_prompt_audio.shape[-1]} samples ≈ {n_frames:.1f} frames")

    mimi.reset_streaming()
    lm_gen.reset_streaming()
    t0 = time.time()
    lm_gen._step_voice_prompt(mimi)  # steps + saves <wav-stem>.pt
    print(f"voice prompt stepped+saved in {time.time()-t0:.1f}s")

    out = os.path.splitext(wav)[0] + ".pt"
    d = torch.load(out, map_location="cpu", weights_only=True)
    print("saved", out)
    print({k: (tuple(v.shape), str(v.dtype)) for k, v in d.items()})


if __name__ == "__main__":
    main()
