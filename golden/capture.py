# Golden-capture oracle for the Qwen3-TTS-12Hz-1.7B-Base -> Burn port.
#
# Runs the official reference (qwen_tts package, transformers 4.57.3) on CPU
# float32 and saves:
#   - quality sample (default sampling)          golden/ref_sample.wav
#   - deterministic greedy run + intermediates   golden/*.npy
# so every Burn stage (speaker encoder, talker backbone, code predictor,
# codec decoder) can be gated against exact tensors.
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/tmp/qwen3-tts-ref")

GOLD = "/tmp/mary-qwen3tts/golden"
MODEL_DIR = "/tmp/qwen3tts-weights/base"
# Reference clip: any ~10 s 24 kHz mono voice clip (e.g. generated with any
# TTS). Goldens are regenerated locally from it and are not committed.
REF_WAV = os.environ.get("REF_VOICE_WAV", "ref_voice.wav")
REF_TEXT = (
    "The tide rolls in across the flat sand, and the evening light settles "
    "slowly over the harbor as the last boats come home."
)  # transcript of REF_WAV -- set to match your clip
TEST_LINE = (
    "If you can hear this clearly, the port worked: the same reference "
    "voice, synthesized end to end by the new engine in real time."
)

def save(name, arr):
    arr = np.asarray(arr)
    np.save(os.path.join(GOLD, name + ".npy"), arr)
    print(f"  saved {name}: {arr.shape} {arr.dtype}")

def main():
    torch.manual_seed(0)
    from qwen_tts import Qwen3TTSModel

    print("loading model (cpu, float32)...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    model = tts.model
    print(f"loaded in {time.time()-t0:.1f}s")

    # ---- prompt build (shared) ----------------------------------------
    audio, sr = sf.read(REF_WAV, dtype="float32")
    assert sr == 24000
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=(audio, sr), ref_text=REF_TEXT, x_vector_only_mode=False
    )
    it = prompt_items[0]
    save("ref_code", it.ref_code.numpy())              # (T,16) int
    save("ref_spk_embedding", it.ref_spk_embedding.float().numpy())  # (2048,)

    # speaker-encoder-only golden: mel input + embedding out
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    mel = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0), n_fft=1024, num_mels=128,
        sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000,
    ).transpose(1, 2)
    save("spk_mel", mel.numpy())                        # (1,T,128)

    # codec-decoder-only golden: decode the ref codes alone
    with torch.inference_mode():
        dec = model.speech_tokenizer.model.decoder
        codes_bkt = it.ref_code.unsqueeze(0).transpose(1, 2)  # (1,16,T)
        # sub-intermediates for debugging
        quant = dec.quantizer.decode(codes_bkt)
        save("codec_quantized", quant.numpy())          # (1,512,T)
        pre = dec.pre_conv(quant).transpose(1, 2)
        pre_t = dec.pre_transformer(inputs_embeds=pre).last_hidden_state
        save("codec_pretransformer", pre_t.numpy())     # (1,T,1024)
        wav_ref_codes = dec(codes_bkt)
        save("codec_refcodes_wav", wav_ref_codes.numpy())  # (1,1,T*1920)

    # ---- text ids ------------------------------------------------------
    input_ids = tts._tokenize_texts([tts._build_assistant_text(TEST_LINE)])
    ref_ids = tts._tokenize_texts([tts._build_ref_text(REF_TEXT)])
    save("text_ids", input_ids[0].numpy())
    save("ref_ids", ref_ids[0].numpy())

    vcp = tts._prompt_items_to_voice_clone_prompt(prompt_items)

    # ---- hooks on the talker -------------------------------------------
    talker_inputs, talker_hiddens, head_logits = [], [], []

    def pre_hook(mod, args, kwargs):
        if len(talker_inputs) < 3 and kwargs.get("inputs_embeds") is not None:
            talker_inputs.append(kwargs["inputs_embeds"].detach().float().numpy())

    def post_hook(mod, args, kwargs, out):
        if len(talker_hiddens) < 3:
            talker_hiddens.append(out.last_hidden_state.detach().float().numpy())

    def head_hook(mod, args, out):
        if len(head_logits) < 3:
            head_logits.append(out.detach().float().numpy())

    h1 = model.talker.model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = model.talker.model.register_forward_hook(post_hook, with_kwargs=True)
    h3 = model.talker.codec_head.register_forward_hook(head_hook)

    # ---- greedy generation (deterministic goldens) ----------------------
    print("greedy generation...")
    t0 = time.time()
    with torch.no_grad():
        codes_list, hidden_list = model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=["English"],
            non_streaming_mode=False,
            do_sample=False,
            top_k=None,
            top_p=None,
            temperature=None,
            subtalker_dosample=False,
            subtalker_top_k=None,
            subtalker_top_p=None,
            subtalker_temperature=None,
            repetition_penalty=1.05,
            max_new_tokens=1200,
        )
    print(f"greedy gen in {time.time()-t0:.1f}s, frames={codes_list[0].shape}")
    h1.remove(); h2.remove(); h3.remove()

    save("prefill_embeds", talker_inputs[0])            # (1,L,2048)
    save("prefill_hidden", talker_hiddens[0])           # (1,L,2048)
    save("prefill_logits", head_logits[0][:, -1:])      # (1,1,3072) last pos
    if len(talker_inputs) > 1:
        save("step1_embeds", talker_inputs[1])
        save("step1_hidden", talker_hiddens[1])
        save("step1_logits", head_logits[1])
    save("greedy_codes", codes_list[0].numpy())          # (T,16)
    save("greedy_talker_hidden", hidden_list[0].float().numpy())  # (T,2048)

    # decode greedy codes (with ref prefix, like the wrapper does)
    with torch.inference_mode():
        full = torch.cat([it.ref_code, codes_list[0]], dim=0)
        wavs, fs = model.speech_tokenizer.decode([{"audio_codes": full}])
        wav = wavs[0]
        cut = int(it.ref_code.shape[0] / full.shape[0] * wav.shape[0])
        sf.write(os.path.join(GOLD, "greedy_full.wav"), wav, fs)
        sf.write(os.path.join(GOLD, "greedy_cut.wav"), wav[cut:], fs)
        save("greedy_wav", wav)
    print("greedy wav saved")

    # ---- quality sample (default sampling params) -----------------------
    print("sampling generation (quality reference)...")
    torch.manual_seed(0)
    t0 = time.time()
    wavs, fs = tts.generate_voice_clone(
        text=TEST_LINE, language="English", voice_clone_prompt=prompt_items
    )
    dt = time.time() - t0
    sf.write(os.path.join(GOLD, "ref_sample.wav"), wavs[0], fs)
    print(f"sampled in {dt:.1f}s -> {len(wavs[0])/fs:.1f}s audio (cpu rtf {len(wavs[0])/fs/dt:.2f})")

    with open(os.path.join(GOLD, "meta.json"), "w") as f:
        json.dump({
            "ref_text": REF_TEXT, "test_line": TEST_LINE,
            "ref_code_frames": int(it.ref_code.shape[0]),
            "greedy_frames": int(codes_list[0].shape[0]),
        }, f, indent=1)
    print("DONE")

if __name__ == "__main__":
    main()
