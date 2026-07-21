# Golden-capture oracle for the Voxtral-Mini-4B-Realtime-2602 -> Burn port.
#
# Runs the HF reference (transformers >= 5.2, voxtral_realtime) on CPU float32,
# deterministic greedy, and saves per-component intermediates + token streams so
# every Burn stage (mel, conv stem, encoder, projector, decoder+ada-norm) can be
# gated against exact tensors.
#
#   .venv_voxtral/bin/python golden/voxtral_capture.py
#
# Outputs under golden/voxtral/:
#   clips/*.wav                    input fixtures — generate locally with any
#                                  TTS (16 kHz mono); NOT committed
#   <clip>_d<N>_tokens.npy         greedy token stream at N delay tokens
#   <clip>_d<N>_text.txt           decoded transcript
#   en_short_* deep intermediates  (see save calls below)
#   meta.json                      shapes, timings, config echo
import json
import os
import time

import numpy as np
import torch

REPO_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voxtral")
CLIPS = ["en_short", "de_short", "denglish", "en_long"]
DELAYS_MS = [480, 960, 2400]  # -> 6, 12, 30 delay tokens
DEEP_CLIP = "en_short"  # component-level intermediates captured on this one
DEEP_DELAY_MS = 480

meta = {"repo_id": REPO_ID, "clips": {}, "delays_ms": DELAYS_MS}


def save(name, arr):
    arr = np.ascontiguousarray(np.asarray(arr))  # mary npy loader is C-order only
    np.save(os.path.join(GOLD, name + ".npy"), arr)
    print(f"  saved {name}: {arr.shape} {arr.dtype}", flush=True)


def load_clip(name):
    import soundfile as sf

    audio, sr = sf.read(os.path.join(GOLD, "clips", name + ".wav"), dtype="float32")
    assert sr == 16000, f"{name}: sr {sr}"
    return audio


def encode_request(tokenizer, audio_array, delay_ms):
    """Mirror VoxtralRealtimeProcessor.__call__ (offline streaming mode) with an
    explicit delay override: returns (input_ids, padded_audio, num_delay_tokens)."""
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.chunk import RawAudio
    from mistral_common.protocol.transcription.request import (
        StreamingMode,
        TranscriptionRequest,
    )

    audio = Audio(audio_array=audio_array, sampling_rate=16000, format="wav")
    req = TranscriptionRequest(
        audio=RawAudio.from_audio(audio),
        streaming=StreamingMode.OFFLINE,
        language=None,
        target_streaming_delay_ms=delay_ms,
    )
    tok = tokenizer.tokenizer.encode_transcription(req)
    audio_cfg = tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config
    n_delay = audio_cfg.get_num_delay_tokens(delay_ms)
    assert len(tok.audios) == 1
    return tok.tokens, tok.audios[0].audio_array, n_delay


def main():
    os.makedirs(GOLD, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("VOXTRAL_THREADS", "8")))

    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    print("loading processor + model (cpu, float32)...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(REPO_ID)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        REPO_ID, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    fe = processor.feature_extractor

    # ---- deep component intermediates on DEEP_CLIP @ DEEP_DELAY_MS ----------
    audio = load_clip(DEEP_CLIP)
    ids, padded_audio, n_delay = encode_request(processor.tokenizer, audio, DEEP_DELAY_MS)
    save("deep_input_ids", np.asarray(ids, dtype=np.float32))  # f32: mary npy loader
    save("deep_padded_audio", padded_audio)

    mel = fe(padded_audio, sampling_rate=16000, center=True, return_tensors="pt")[
        "input_features"
    ]  # (1, 128, T_mel)
    save("deep_mel", mel.numpy())

    with torch.inference_mode():
        # conv stem
        stem = model.model.audio_tower.embedder(mel)  # (1, T_enc, 1280)
        save("deep_conv_stem", stem.numpy())

        # full encoder (single pass, no cache) + per-layer taps
        enc = model.model.audio_tower(
            input_features=mel, output_hidden_states=True, return_dict=True
        )
        hs = enc.hidden_states  # tuple: embed + each layer
        save("deep_enc_embed", hs[0].numpy())
        for i in (0, 15, 31):
            save(f"deep_enc_layer{i}", hs[i + 1].numpy())
        save("deep_enc_final", enc.last_hidden_state.numpy())

        # projector
        ds = model.config.downsample_factor
        proj_in = enc.last_hidden_state.reshape(1, -1, 1280 * ds)
        audio_embeds = model.model.multi_modal_projector(proj_in)
        save("deep_audio_embeds", audio_embeds.numpy())

        # time embedding + per-layer ada scales for all gate delays
        for dms in DELAYS_MS:
            nd = {480: 6, 960: 12, 2400: 30}[dms]
            t_cond = model.model.time_embedding(
                torch.full((1,), nd, dtype=torch.float32)
            )
            save(f"t_cond_d{nd}", t_cond.numpy())
            scales = [
                (1 + layer.ada_rms_norm(t_cond)).numpy()
                for layer in model.model.language_model.layers
            ]
            save(f"ada_scales_d{nd}", np.stack(scales))

        # decoder prefill: token prompt + aligned audio embeds, per-layer taps
        L = len(ids)
        input_ids = torch.tensor([ids], dtype=torch.long)
        tok_embeds = model.model.get_input_embeddings()(input_ids)
        prefill_embeds = tok_embeds + audio_embeds[:, :L, :]
        save("deep_prefill_embeds", prefill_embeds.numpy())
        dec = model.model.language_model(
            inputs_embeds=prefill_embeds,
            t_cond=model.model.time_embedding(
                torch.full((1,), n_delay, dtype=torch.float32)
            )[None, ...],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        dhs = dec.hidden_states
        for i in (0, 12, 25):
            save(f"deep_dec_layer{i}", dhs[i + 1].numpy())
        save("deep_dec_final", dec.last_hidden_state.numpy())
        logits = model.lm_head(dec.last_hidden_state[:, -1:, :])
        save("deep_prefill_logits", logits.numpy())

    # ---- greedy token streams for every clip x delay -------------------------
    for clip in CLIPS:
        audio = load_clip(clip)
        meta["clips"][clip] = {"num_samples": int(audio.shape[0])}
        for dms in DELAYS_MS:
            ids, padded_audio, n_delay = encode_request(
                processor.tokenizer, audio, dms
            )
            mel = fe(padded_audio, sampling_rate=16000, center=True, return_tensors="pt")[
                "input_features"
            ]
            input_ids = torch.tensor([ids], dtype=torch.long)
            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    input_features=mel,
                    attention_mask=torch.ones_like(input_ids),
                    num_delay_tokens=n_delay,
                    do_sample=False,
                )
            dt = time.time() - t0
            tokens = out[0].numpy().astype(np.float32)  # f32: mary npy loader
            text = processor.tokenizer.decode(
                out[0].tolist(), skip_special_tokens=True
            )
            save(f"{clip}_d{n_delay}_tokens", tokens)
            with open(os.path.join(GOLD, f"{clip}_d{n_delay}_text.txt"), "w") as f:
                f.write(text)
            meta["clips"][clip][f"d{n_delay}"] = {
                "prompt_len": len(ids),
                "mel_frames": int(mel.shape[-1]),
                "gen_len": int(tokens.shape[0]),
                "seconds": round(dt, 1),
                "text": text,
            }
            print(f"[{clip} d={n_delay}] {dt:.0f}s :: {text!r}", flush=True)

    with open(os.path.join(GOLD, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
