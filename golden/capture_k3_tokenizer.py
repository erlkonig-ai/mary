# Kimi-K3 tiktoken tokenizer golden battery: the SHIPPED Python implementation
# (`tokenization_kimi.TikTokenTokenizer`, which drives the `tiktoken` package)
# run over a battery of adversarial strings + a deterministic fuzz corpus,
# dumped as JSON for `k3_tokenizer_gate` to gate mary's pure-Rust port against.
#
# The gate compares full ID SEQUENCES, so this file is the only source of
# truth for what "correct" means. Text is carried as hex of its UTF-8 bytes
# (`text_hex`) so no JSON/shell escaping can drift between the two sides.
#
#   python golden/capture_k3_tokenizer.py
import json
import os
import random
import sys

MODEL_DIR = os.environ.get("K3_DIR", "./kimi-k3")
OUT_DIR = os.environ.get("K3_GOLD", "/tmp/mary-k3tok/golden")

sys.path.insert(0, MODEL_DIR)
from tokenization_kimi import TikTokenTokenizer  # noqa: E402

from tokenizers import AddedToken  # noqa: E402


def load_shipped():
    """Instantiate the shipped tokenizer exactly as AutoTokenizer would."""
    cfg = json.load(open(os.path.join(MODEL_DIR, "tokenizer_config.json")))
    added = {
        int(i): AddedToken(
            v["content"],
            lstrip=v["lstrip"],
            rstrip=v["rstrip"],
            normalized=v["normalized"],
            single_word=v["single_word"],
            special=v["special"],
        )
        for i, v in cfg["added_tokens_decoder"].items()
    }
    return TikTokenTokenizer(
        vocab_file=os.path.join(MODEL_DIR, "tiktoken.model"),
        bos_token=cfg["bos_token"],
        eos_token=cfg["eos_token"],
        unk_token=cfg["unk_token"],
        pad_token=cfg["pad_token"],
        additional_special_tokens=cfg["additional_special_tokens"],
        added_tokens_decoder=added,
    )


def fnv1a64(chunks):
    h = 0xCBF29CE484222325
    for c in chunks:
        for b in c:
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def vocab_fingerprint(path):
    """FNV-1a/64 over (len|bytes|rank) of every line, in file order. The Rust
    parser computes the same thing; equality proves both sides parsed all
    163,584 base64 lines to identical (bytes, rank) pairs."""
    import base64

    n = 0
    chunks = []
    for line in open(path, "rb"):
        line = line.strip()
        if not line:
            continue
        tok, rank = line.split()
        b = base64.b64decode(tok)
        r = int(rank)
        chunks.append(len(b).to_bytes(4, "little"))
        chunks.append(b)
        chunks.append(r.to_bytes(4, "little"))
        n += 1
    return n, fnv1a64(chunks)


# ── the named battery. Every entry is (name, text, allow_special). ──
def battery():
    cases = []

    def add(name, text, allow_special=True):
        cases.append((name, text, allow_special))

    # empty + degenerate
    add("empty", "")
    add("single_space", " ")
    add("single_char", "a")
    add("single_newline", "\n")

    # plain ASCII prose
    add("ascii_prose", "The quick brown fox jumps over the lazy dog.")
    add(
        "ascii_prose_long",
        "Tokenization is the process of splitting text into subword units. "
        "A byte-level BPE tokenizer first pre-tokenizes with a regex, then "
        "merges adjacent pairs by rank until no merge applies.",
    )
    add("ascii_punct", "punctuation!?;:()[]{}#@$%^&*~`|\\/<>=+-_")
    add("url", "https://github.com/triblespace/faculties?tab=readme#usage")

    # CJK — the dedicated \p{Han} branch
    add("han_only", "你好世界")
    add("han_sentence", "机器学习是人工智能的一个分支。")
    add("han_long", "量子计算机使用量子比特进行计算，它们可以同时处于多个状态。")
    add("han_latin_adjacent", "中文abc中文123中文")
    add("han_with_apostrophe", "中文's 中文'T")
    add("han_punct_mix", "你好，世界！这是一个测试。")
    add("kana", "日本語のテキストと漢字")
    add("hangul", "안녕하세요 세계")

    # mixed scripts
    add(
        "mixed_scripts",
        "Hello 世界 Привет κόσμος مرحبا שלום こんにちは",
    )
    add("german", "Größe, Straße, äöüß — German umlauts")
    add("cyrillic", "Привет, мир! Как дела?")
    add("smart_quotes", "Lighthouse — “a light by the sea”, with ‘smart’ quotes and… ellipsis")

    # contractions — the (?i:'s|'t|'re|'ve|'m|'ll|'d) group
    add("contractions_lower", "don't it's we're I've I'm they'll he'd")
    add("contractions_upper", "DON'T IT'S WE'RE I'VE I'M THEY'LL HE'D")
    add("contractions_mixed_case", "Don'T iT'S We'Re i'Ve I'm tHeY'lL hE'd")
    add("contraction_after_upper_run", "ABC'S ABC'T ABC'RE ABC'VE ABC'M ABC'LL ABC'D")
    add("contraction_bare", "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D")
    add("contraction_not_a_suffix", "don'x it'q we'zz")
    add("apostrophe_unicode", "don’t it’s")  # U+2019, NOT in the group

    # digits — \p{N}{1,3}
    add("digits_run", "1234")
    add("digits_various", "0 7 42 007 1234567890 3.14159 -42 1e-8")
    add("digits_long", "123456789012345678901234567890")
    add("digits_date", "2026-08-05T13:40:02Z")
    add("digits_nonascii", "١٢٣٤ 一二三 ⅠⅡ")

    # whitespace
    add("leading_space", "   leading spaces")
    add("trailing_space", "trailing spaces   ")
    add("only_spaces", "     ")
    add("interior_runs", "a   b\tc\t\td")
    add("newlines", "line one\nline two\n\nline four")
    add("crlf", "a\r\nb\rc\n\r\nd")
    add("newline_runs", "\n\n\n\n")
    add("mixed_ws_newline", "  \n  \t \n\n   ")
    add("trailing_ws_eos", "word   ")
    add("trailing_ws_then_char", "word   x")
    add("leading_ws_only_tab", "\t\t\ttabbed")
    add("nbsp", "a b  c")
    add("unicode_ws", "a b c　d e")

    # emoji / multi-byte UTF-8 / byte fallback
    add("emoji", "emoji \U0001f30a\U0001f525\U0001f916 here")
    add("emoji_zwj", "\U0001f469‍\U0001f469‍\U0001f467‍\U0001f466 family")
    add("emoji_flags", "\U0001f1e9\U0001f1ea \U0001f1ef\U0001f1f5 \U0001f1fa\U0001f1f8")
    add("emoji_skin_tone", "\U0001f44d\U0001f3fd \U0001f44b\U0001f3ff")
    add("combining", "é vs é, äöü")
    add("math_symbols", "Ω≈ç√∫˜µ≤≥÷")
    add("rare_planes", "\U0001d49c\U0001d4b7 \U000200b9 \U0002a6a5 \U0001f0a1")
    add("control_chars", "a\x00b\x01c\x7fd\x1be")

    # special-token markup, both ways
    add("special_bos_eos", "[BOS]hello world[EOS]", True)
    add("special_bos_eos_plain", "[BOS]hello world[EOS]", False)
    add("special_end_of_msg", "before<|end_of_msg|>after", True)
    add("special_end_of_msg_plain", "before<|end_of_msg|>after", False)
    add("special_headers", "[start_header_id]user[end_header_id]hi[EOT]", True)
    add("special_headers_plain", "[start_header_id]user[end_header_id]hi[EOT]", False)
    add("special_media", "<|media_begin|>img<|media_content|>x<|media_end|><|media_pad|>", True)
    add("special_reserved", "a<|reserved_token_163600|>b<|reserved_token_163839|>c", True)
    add("special_reserved_plain", "a<|reserved_token_163600|>b<|reserved_token_163839|>c", False)
    add("special_osagent", "<osagent_mode>on", True)
    add("special_open_close_sep", "<|open|>a<|sep|>b<|close|>", True)
    add("special_adjacent", "[BOS][BOS][EOS][EOS]", True)
    add("special_only", "[BOS]", True)
    add("special_near_miss", "<|not_a_real_token|> [NOPE] <|im_end|>", True)
    add("special_unclosed", "<|end_of_msg incomplete", True)
    add("special_no_space_han", "中文<|end_of_msg|>中文", True)

    # the shipped wrapper's >25k-char chunking (_split_whitespaces_or_nonwhitespaces)
    add("long_nonws_run", "a" * 30000)
    # 25000 % 7 == 3, so the chunk boundary lands mid-unit and the seam is
    # visible in the ids — unlike a run of one repeated character, where the
    # split happens to fall on a token boundary and hides the chunker.
    add("long_nonws_boundary", "abcdefg" * 4000)
    add("long_ws_run", " " * 30000 + "end")
    add("long_alternating", ("word " * 8000))

    # real source text (whatever is on disk, so the corpus isn't hand-picked)
    for fname in ("tokenization_kimi.py", "encoding_k3.py", "README.md"):
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            txt = open(p, encoding="utf-8", errors="replace").read()
            add("file_" + fname.replace(".", "_"), txt[:20000], False)

    return cases


HAN = "一二三四五中文字机器学习量子计算你好世界国家"
POOLS = [
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    " \t\n\r 　",
    "!?.,;:'\"()[]{}<>|/\\-_=+*&^%$#@~`",
    HAN,
    "あいうえおアイウエオ",
    "абвгдαβγδ",
    "\U0001f600\U0001f30a\U0001f525\U0001f916\U0001f44d‍\U0001f3fd",
    "̧́̈",
    "äöüßéèêŁł",
    "\x00\x01\x1b\x7f",
]
FRAGMENTS = [
    "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
    "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D",
    "[BOS]", "[EOS]", "<|end_of_msg|>", "[EOT]", "<|media_pad|>",
    "<|reserved_token_163700|>", "<osagent_mode>", "[start_header_id]",
    "  ", "\n\n", " \n", "\n ", "the ", " the", "ing", "tion", "1234", "007",
]


def fuzz_corpus(n=3000, seed=0xC0FFEE):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(0, 12)):
            if rng.random() < 0.25:
                parts.append(rng.choice(FRAGMENTS))
            else:
                pool = rng.choice(POOLS)
                parts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 8))))
        s = "".join(parts)
        out.append((s, rng.random() < 0.5))
    return out


def main():
    tok = load_shipped()
    n, fp = vocab_fingerprint(os.path.join(MODEL_DIR, "tiktoken.model"))
    print(f"vocab: {n} entries, fnv1a64=0x{fp:016x}, n_words={tok.n_words}")

    specials = sorted(((name, i) for name, i in tok.special_tokens.items()), key=lambda x: x[1])

    def case(name, text, allow_special):
        ids = tok.encode(text, allow_special_tokens=allow_special)
        dec = tok.decode(ids)
        return {
            "name": name,
            "text": text,
            "text_hex": text.encode("utf-8").hex(),
            "allow_special": allow_special,
            "ids": ids,
            "decoded_hex": dec.encode("utf-8").hex(),
        }

    cases = [case(n_, t, a) for (n_, t, a) in battery()]
    fuzz = [
        {"text_hex": s.encode("utf-8").hex(),
         "allow_special": a,
         "ids": tok.encode(s, allow_special_tokens=a)}
        for (s, a) in fuzz_corpus()
    ]

    doc = {
        "model_dir": MODEL_DIR,
        "n_vocab": tok.n_words,
        "n_base": n,
        "vocab_fnv1a64": f"{fp:016x}",
        "pat_str": tok.pat_str,
        "num_reserved_special_tokens": tok.num_reserved_special_tokens,
        "special_tokens": [{"name": s, "id": i} for s, i in specials],
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "pad_id": tok.pad_id,
        "unk_id": tok.unk_id,
        "cases": cases,
        "fuzz": fuzz,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "k3_tokenizer_battery.json")
    with open(out, "w") as f:
        json.dump(doc, f, ensure_ascii=False)
    tot = sum(len(c["ids"]) for c in cases) + sum(len(c["ids"]) for c in fuzz)
    print(f"saved {out}: {len(cases)} named cases + {len(fuzz)} fuzz strings, {tot} tokens")
    empties = [c["name"] for c in cases if not c["ids"] and c["text_hex"]]
    print(f"non-empty text with empty ids (should be none): {empties}")


if __name__ == "__main__":
    main()
