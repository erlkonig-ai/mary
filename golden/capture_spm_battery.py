# SPM tokenizer battery golden for the Phase-5 prompt machinery: the oracle
# venv's sentencepiece tokenization of a battery of strings (ASCII prompts,
# whitespace edge cases, German/CJK/Cyrillic, emoji -> byte fallback, the
# metasymbol itself), dumped as JSON for `personaplex_probe prompt` to gate
# mary's pure-Rust unigram implementation against.
#
#   /tmp/personaplex_scratch/moshi_venv/bin/python golden/capture_spm_battery.py
import json
import os

import sentencepiece

GOLD = "/tmp/mary-personaplex/golden"
MODEL = "/tmp/personaplex_scratch/ckpt/tokenizer_spm_32k_3.model"

BATTERY = [
    # the capture's exact wrapped system prompt (also gated vs text_prompt_tokens.npy)
    "<system> You are a wise and friendly teacher. Answer questions or provide "
    "advice in a clear and engaging way. <system>",
    "",
    " ",
    "a",
    "Hello",
    "Hello world",
    " leading space",
    "trailing space ",
    "double  space and triple   space",
    "tabs\tand\nnewlines\r\nmixed",
    "<system> nested <system> tags <system>",
    "You are a careful archivist. Your memory is the pile; keep it exact.",
    "Lighthouse — “a light by the sea”, with ‘smart’ quotes and… ellipsis",
    "Größe, Straße, äöüß — German umlauts",
    "日本語のテキストと漢字",
    "안녕하세요 세계",
    "Привет, мир!",
    "emoji 🌊🔥🤖 and combining é vs é",
    "1234567890 3.14159 -42 1e-8",
    "punctuation!?;:()[]{}#@$%^&*",
    "▁ the metasymbol U+2581 itself ▁▁",
    "CamelCase MiXeD UPPER lower",
    "https://github.com/triblespace/faculties?tab=readme#usage",
    "Ω≈ç√∫˜µ≤≥÷ mathematical symbols",
    "one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty",
]


def main():
    sp = sentencepiece.SentencePieceProcessor(MODEL)
    battery = [{"text": t, "ids": sp.encode(t)} for t in BATTERY]
    os.makedirs(GOLD, exist_ok=True)
    out = os.path.join(GOLD, "spm_battery.json")
    with open(out, "w") as f:
        json.dump(battery, f, ensure_ascii=False, indent=1)
    print(f"saved {out}: {len(battery)} strings, "
          f"{sum(len(b['ids']) for b in battery)} tokens")


if __name__ == "__main__":
    main()
