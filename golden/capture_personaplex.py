# Golden-capture oracle for the PersonaPlex-7B (Moshi) -> mary port, Phase 0.
#
# Runs the NVIDIA reference (the PATCHED moshi package from
# github.com/NVIDIA/personaplex -- NOT PyPI moshi) on CPU float32,
# deterministically (greedy, fixed seed, fixed inputs), and dumps .npy goldens
# for every surface the Rust LM port will be gated against:
#
#   (a) temporal transformer per-step hidden state + text logits
#       (`tt_hidden.npy` [S,4096], `tt_text_logits.npy` [S,32000]) -- the
#       "prefill hidden + last-position logits" are rows `phases[...]-1`,
#   (b) depth transformer per-codebook logits + greedy argmax for every step
#       (`dep_logits.npy` [S,16,2048], `dep_tokens.npy` [S,16]),
#   (c) N greedy frames of full token output (`out_tokens.npy` [N,17] --
#       delay-adjusted LMGen output, order [text, agent 1..8, user 1..8]),
#   (d) the full prefill/step token stream (`step_tokens.npy` [S_tok,17], the
#       exact `input_` fed to forward_codes each step; the voice-prompt phase
#       feeds EMBEDDINGS instead -- `vp_embeddings.npy` -- via
#       forward_embeddings, and `phases.json` says which step is which),
#   (e) the decoded output AUDIO for the same N output frames
#       (`out_audio.npy` [N*1920] f32): agent streams 1..9 of each undelayed
#       out frame through the STREAMING Mimi decoder, per frame, exactly
#       mirroring offline.py's decode_tokens_to_pcm (the production path).
#       `out_audio_batch.npy` is the same 25x8 code matrix through a fresh
#       NON-streaming batch decode -- the cross-check that isolates
#       streaming-vs-batch decoder differences from LM/wiring differences.
#
# Flow mirrors moshi/offline.py (voice prompt -> 0.5 s silence -> text prompt
# -> 0.5 s silence -> user audio), EXCEPT the warmup loop is skipped: it only
# primes CUDA graphs (no-op on CPU) and would prepend 4 junk frames to the
# streaming state. Voice prompt is a packaged `.pt` (pre-saved embeddings +
# cache replay -- the production path), so Mimi never encodes the voice.
#
# Run from the moshi venv:
#   /tmp/personaplex_scratch/moshi_venv/bin/python golden/capture_personaplex.py
import json
import os
import time

import numpy as np
import torch

GOLD = "/tmp/mary-personaplex/golden"
CKPT = "/tmp/personaplex_scratch/ckpt"
VOICE_PT = "/tmp/personaplex_scratch/voices/NATM0.pt"
INPUT_WAV = os.environ.get("REF_VOICE_WAV", "ref_voice.wav")  # user-side audio (any voice clip)
INPUT_SECONDS = 2.0  # 25 frames at 12.5 Hz
SEED = 42424242
TEXT_PROMPT = (
    "You are a wise and friendly teacher. Answer questions or provide advice "
    "in a clear and engaging way."
)


def save(name, arr):
    arr = np.asarray(arr)
    np.save(os.path.join(GOLD, name + ".npy"), arr)
    print(f"  saved {name}: {arr.shape} {arr.dtype}")


def main():
    os.makedirs(GOLD, exist_ok=True)
    from moshi.models import loaders, LMGen
    from moshi.models.lm import load_audio, _iterate_audio, encode_from_sphn
    from moshi.offline import seed_all, wrap_with_system_tags
    import sentencepiece

    seed_all(SEED)

    t0 = time.time()
    print("loading mimi (cpu)...")
    mimi = loaders.get_mimi(os.path.join(CKPT, loaders.MIMI_NAME), "cpu")
    print(f"mimi loaded in {time.time()-t0:.1f}s")

    tok = sentencepiece.SentencePieceProcessor(
        os.path.join(CKPT, loaders.TEXT_TOKENIZER_NAME)
    )

    t0 = time.time()
    print("loading moshi LM (cpu, float32)...")
    lm = loaders.get_moshi_lm(
        os.path.join(CKPT, loaders.MOSHI_NAME), device="cpu", dtype=torch.float32
    )
    lm.eval()
    load_secs = time.time() - t0
    print(f"moshi LM loaded in {load_secs:.1f}s")

    # ---- instrument the LM (before LMGen captures the methods) -------------
    # NOTE: forward_codes(input_) == forward_embeddings(embed_codes(input_)),
    # so wrapping forward_embeddings ALONE sees every temporal step exactly
    # once (both the token-fed steps and the voice-prompt embedding-replay
    # steps). The forward_codes wrapper only records which steps were
    # token-fed, and with which tokens.
    step_tokens = []        # input_ [1,17,1] per forward_codes call
    step_token_idx = []     # temporal-step index each step_tokens row lands on
    tt_hidden = []          # transformer_out [1,1,4096] per temporal step
    tt_text_logits = []     # text_logits [1,1,1,32000] per temporal step

    orig_fc = lm.forward_codes
    orig_fe = lm.forward_embeddings

    def fc(input_):
        step_token_idx.append(len(tt_hidden))
        step_tokens.append(input_.detach().cpu().numpy().copy())
        return orig_fc(input_)

    def fe(embeddings):
        out, tl = orig_fe(embeddings)
        tt_hidden.append(out.detach().float().cpu().numpy().copy())
        tt_text_logits.append(tl.detach().float().cpu().numpy().copy())
        return out, tl

    lm.forward_codes = fc
    lm.forward_embeddings = fe

    frame_rate = 12.5
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * frame_rate),
        sample_rate=mimi.sample_rate,
        device="cpu",
        frame_rate=frame_rate,
        use_sampling=False,  # greedy: argmax everywhere, fully deterministic
        return_logits=True,
    )

    dep_in_text = []   # next_text_token fed to the depformer per step
    dep_tokens = []    # greedy tokens [1,16]
    dep_logits = []    # per-codebook logits [1,16,2048]
    orig_ds = lm_gen.depformer_step

    def ds(text_token, transformer_out, audio_tokens, audio_provided):
        toks, logits = orig_ds(text_token, transformer_out, audio_tokens, audio_provided)
        dep_in_text.append(text_token.detach().cpu().numpy().copy())
        dep_tokens.append(toks.detach().cpu().numpy().copy())
        dep_logits.append(logits.detach().float().cpu().numpy().copy())
        return toks, logits

    lm_gen.depformer_step = ds  # instance attr shadows the class method

    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # ---- prompts (NO warmup -- see header) ---------------------------------
    # NOTE: lm_gen.load_voice_prompt_embeddings() calls torch.load without
    # map_location; the packaged voices were saved from CUDA, so on a
    # CPU-only box it raises. Load manually (same semantics + map_location).
    vp = torch.load(VOICE_PT, map_location="cpu")
    lm_gen.voice_prompt = VOICE_PT
    lm_gen.voice_prompt_audio = None
    lm_gen.voice_prompt_embeddings = vp["embeddings"].to(torch.float32)
    lm_gen.voice_prompt_cache = vp["cache"]
    save("vp_embeddings", vp["embeddings"].float().numpy())
    save("vp_cache", vp["cache"].numpy())
    lm_gen.text_prompt_tokens = tok.encode(wrap_with_system_tags(TEXT_PROMPT))
    save("text_prompt_tokens", np.asarray(lm_gen.text_prompt_tokens, dtype=np.int64))

    mimi.reset_streaming()
    lm_gen.reset_streaming()

    phases = {}
    t0 = time.time()
    lm_gen._step_voice_prompt(mimi)
    phases["after_voice"] = len(tt_hidden)
    lm_gen._step_audio_silence()
    phases["after_silence1"] = len(tt_hidden)
    lm_gen._step_text_prompt()
    phases["after_text"] = len(tt_hidden)
    lm_gen._step_audio_silence()
    phases["after_silence2"] = len(tt_hidden)
    prompt_secs = time.time() - t0
    print(f"prompt phases done in {prompt_secs:.1f}s ({len(tt_hidden)} temporal steps)")
    mimi.reset_streaming()

    # ---- user audio: encode with Mimi, step the LM greedily ----------------
    user_audio = load_audio(INPUT_WAV, mimi.sample_rate)  # numpy (C, T)
    n = int(INPUT_SECONDS * mimi.sample_rate)
    user_audio = user_audio[..., :n]
    save("user_audio", user_audio)

    out_tokens = []
    out_pcm = []  # streaming per-frame Mimi decode of agent streams 1..9
    user_codes = []
    t0 = time.time()
    with torch.no_grad():
        for user_encoded in encode_from_sphn(
            mimi,
            _iterate_audio(user_audio, sample_interval_size=lm_gen._frame_size, pad=True),
            max_batch=1,
        ):
            for c in range(user_encoded.shape[-1]):
                step_in = user_encoded[:, :, c : c + 1]
                user_codes.append(step_in.detach().cpu().numpy().copy())
                out, _logits = lm_gen.step(input_tokens=step_in)
                if out is not None:
                    out_tokens.append(out.detach().cpu().numpy().copy())
                    # offline.py decode_tokens_to_pcm: agent audio = out[:, 1:9],
                    # through the SAME streaming mimi (decode state is fresh at
                    # the first gen frame -- reset_streaming ran after prompts).
                    pcm = mimi.decode(out[:, 1:9])
                    out_pcm.append(pcm.detach().cpu().numpy()[0, 0].copy())
    gen_secs = time.time() - t0
    total_steps = len(tt_hidden)
    gen_steps = total_steps - phases["after_silence2"]
    print(
        f"user-audio phase: {gen_steps} steps in {gen_secs:.1f}s "
        f"({gen_secs/max(gen_steps,1):.2f} s/step)"
    )

    # ---- dump ---------------------------------------------------------------
    save("step_tokens", np.concatenate(step_tokens, axis=0)[:, :, 0] if step_tokens else np.zeros((0, 17)))
    save("step_token_idx", np.asarray(step_token_idx, dtype=np.int64))
    save("tt_hidden", np.concatenate([h.reshape(1, -1) for h in tt_hidden], axis=0))
    save("tt_text_logits", np.concatenate([t.reshape(1, -1) for t in tt_text_logits], axis=0))
    save("dep_in_text", np.concatenate(dep_in_text, axis=0))
    save("dep_tokens", np.concatenate(dep_tokens, axis=0))
    save("dep_logits", np.concatenate(dep_logits, axis=0))
    save("out_tokens", np.concatenate(out_tokens, axis=0)[:, :, 0])
    save("user_codes", np.concatenate(user_codes, axis=0)[:, :, 0])
    save("out_audio", np.concatenate(out_pcm))

    # Batch cross-check: the same agent-code matrix through a FRESH mimi with
    # no streaming state (full-sequence decode). Isolates streaming-vs-batch
    # decoder deltas from LM/wiring deltas on the Rust side.
    all_out = np.concatenate(out_tokens, axis=2)  # [1, 17, N]
    with torch.no_grad():
        mimi_batch = loaders.get_mimi(os.path.join(CKPT, loaders.MIMI_NAME), "cpu")
        pcm_batch = mimi_batch.decode(torch.from_numpy(all_out[:, 1:9, :]))
    save("out_audio_batch", pcm_batch.detach().cpu().numpy()[0, 0])

    meta = {
        "seed": SEED,
        "greedy": True,
        "dtype": "float32",
        "device": "cpu",
        "warmup_skipped": True,
        "voice_prompt": os.path.basename(VOICE_PT),
        "text_prompt": TEXT_PROMPT,
        "input_wav": INPUT_WAV,
        "input_seconds": INPUT_SECONDS,
        "phases": phases,
        "n_token_steps": len(step_token_idx),
        "total_temporal_steps": total_steps,
        "lm_load_secs": round(load_secs, 1),
        "prompt_secs": round(prompt_secs, 1),
        "gen_secs": round(gen_secs, 1),
        "delays": lm.delays,
    }
    with open(os.path.join(GOLD, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("meta:", json.dumps(meta, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
