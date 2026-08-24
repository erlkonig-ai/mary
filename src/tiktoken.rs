//! mary — tiktoken-style byte-level BPE (the Kimi-K3 tokenizer), in Rust.
//!
//! The engine half of the tiktoken lane; [`crate::tokenizer`] holds the graph
//! half (`ty::TIKTOKEN`, `attrs::piece_bytes`, …). Unlike a HuggingFace BPE
//! there is **no merges list**: the rank table *is* the merge order — a pair
//! may merge iff its concatenation is a token, and its rank is that token's id.
//! So `tiktoken.model` (one `<base64 token bytes> <rank>` per line) is the
//! whole model, and encoding is:
//!
//!   1. pre-tokenize with `pat_str` (a fancy-regex `find_iter`, leftmost-first),
//!   2. per piece: if the piece is itself a token, emit it; otherwise merge the
//!      adjacent pair of lowest rank, repeatedly, until none applies.
//!
//! Ported against the shipped `tokenization_kimi.py` and gated id-for-id by
//! `src/bin/k3_tokenizer_gate.rs` (goldens from `golden/capture_k3_tokenizer.py`).
//!
//! **Why `fancy-regex` and not `regex`.** `pat_str` uses three non-trivial
//! constructs: `&&` character-class intersection, `(?i:…)` inline-case groups,
//! and — the blocker — a negative lookahead `\s+(?!\S)`. The `regex` crate
//! accepts the first two and *rejects the third* ("look-around, including
//! look-ahead and look-behind, is not supported"), and there is no lookahead-
//! free rewrite: that branch means "the whitespace run, minus its last
//! character, unless the run ends the haystack", which is exactly the
//! backtracking behaviour `\s+` + `(?!\S)` produces and which no finite
//! automaton alternation expresses. `fancy-regex` is also literally the engine
//! `tiktoken` itself runs the pattern on, so the pre-tokenizer is identical by
//! construction rather than by argument. (Observable difference, from the
//! goldens: `"word   "` → `["word", "   "]` but `"word   x"` → `["word", "  ",
//! " x"]`.)
//!
//! Byte-level, so token bytes are arbitrary — 1,172 of Kimi-K3's 163,584 base
//! tokens are not valid UTF-8 (mid-codepoint merge fragments). Everything here
//! is `[u8]`; `String` appears only at the API edges.

use std::collections::HashMap;

use fancy_regex::Regex;

type Err = Box<dyn std::error::Error>;

/// A token's rank == its id in the base vocab. `u32` matches tiktoken's `Rank`.
pub type Rank = u32;

/// Sentinel for "this adjacent pair does not merge" (tiktoken's `Rank::MAX`).
const NO_MERGE: Rank = Rank::MAX;

/// Kimi-K3's pre-tokenizer pattern, verbatim from `TikTokenTokenizer.pat_str`
/// in the shipped `tokenization_kimi.py` (the `"|".join([...])` of eight
/// branches, in order). The gate asserts this equals the pattern the shipped
/// Python reports, so a model-side change to `pat_str` breaks the build's
/// trust rather than silently shifting ids.
pub const KIMI_K3_PAT_STR: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// The shipped wrapper's chunking constants (`_encode_text_piece`). Both are
/// *character* counts, not byte counts, and both split the input before it ever
/// reaches the regex — so they can change the tokenization at the seam. Ported
/// because they are part of the shipped `encode`, not of tiktoken proper.
const TIKTOKEN_MAX_ENCODE_CHARS: usize = 400_000;
const MAX_NO_WHITESPACES_CHARS: usize = 25_000;

// ═══════════════════════════════════════════════════════════════════════════
// tiktoken.model — `<base64 token bytes> <rank>` per line
// ═══════════════════════════════════════════════════════════════════════════

/// Decode one standard-alphabet base64 group. Strict: rejects any character
/// outside the alphabet, any interior padding, and any length ≢ 0 (mod 4).
/// (Hand-rolled rather than pulling a `base64` dep for ~30 lines; the gate's
/// vocab fingerprint re-derives all 163,584 entries against Python's
/// `base64.b64decode`, so a decoder bug cannot pass unnoticed.)
fn b64_decode(s: &[u8]) -> Result<Vec<u8>, Err> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    if s.len() % 4 != 0 {
        return Err(format!("base64 length {} not a multiple of 4", s.len()).into());
    }
    let pad = s.iter().rev().take_while(|&&c| c == b'=').count();
    if pad > 2 {
        return Err("base64 has more than 2 padding bytes".into());
    }
    let body = &s[..s.len() - pad];
    let mut out = Vec::with_capacity(body.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &c in body {
        let v = val(c).ok_or_else(|| format!("invalid base64 byte {c:#04x}"))? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    // Leftover bits must be zero, else the encoding was non-canonical.
    if acc & ((1 << bits) - 1) != 0 {
        return Err("base64 has non-zero trailing bits".into());
    }
    Ok(out)
}

/// Parse a `tiktoken.model` file into `(token bytes, rank)` pairs, in file
/// order. The dual of tiktoken's `load_tiktoken_bpe`.
pub fn parse_tiktoken_model(bytes: &[u8]) -> Result<Vec<(Vec<u8>, Rank)>, Err> {
    let mut out = Vec::new();
    for (lineno, line) in bytes.split(|&b| b == b'\n').enumerate() {
        let line = match line.strip_suffix(b"\r") {
            Some(l) => l,
            None => line,
        };
        if line.is_empty() {
            continue;
        }
        let sp = line
            .iter()
            .position(|&b| b == b' ')
            .ok_or_else(|| format!("line {}: no space separator", lineno + 1))?;
        let tok = b64_decode(&line[..sp]).map_err(|e| format!("line {}: {e}", lineno + 1))?;
        let rank: Rank = std::str::from_utf8(&line[sp + 1..])?
            .trim()
            .parse()
            .map_err(|e| format!("line {}: bad rank: {e}", lineno + 1))?;
        out.push((tok, rank));
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// the encoder
// ═══════════════════════════════════════════════════════════════════════════

/// A tiktoken byte-level BPE encoding: the rank table, the special-token table,
/// and the compiled pre-tokenizer. Construct with [`Tiktoken::new`].
pub struct Tiktoken {
    ranks: HashMap<Vec<u8>, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_encoder: HashMap<String, Rank>,
    special_decoder: HashMap<Rank, String>,
    pat: Regex,
    /// Alternation of every special token, longest-first — the "next special
    /// token from here" scanner (tiktoken's `special_regex`).
    special_pat: Option<Regex>,
}

impl Tiktoken {
    /// Build from the parsed rank table, the special-token table (`name → id`),
    /// and a pre-tokenizer pattern. Errors if a byte value 0..=255 has no
    /// single-byte token — without those, byte-fallback is impossible and BPE
    /// would panic mid-encode (Kimi-K3 has all 256).
    pub fn new(
        ranks: impl IntoIterator<Item = (Vec<u8>, Rank)>,
        specials: impl IntoIterator<Item = (String, Rank)>,
        pat_str: &str,
    ) -> Result<Self, Err> {
        let ranks: HashMap<Vec<u8>, Rank> = ranks.into_iter().collect();
        for b in 0u16..=255 {
            if !ranks.contains_key(&vec![b as u8]) {
                return Err(format!("rank table has no single-byte token for {b:#04x}").into());
            }
        }
        let decoder: HashMap<Rank, Vec<u8>> = ranks.iter().map(|(k, &v)| (v, k.clone())).collect();
        if decoder.len() != ranks.len() {
            return Err("rank table has duplicate ranks".into());
        }
        let special_encoder: HashMap<String, Rank> = specials.into_iter().collect();
        let special_decoder: HashMap<Rank, String> = special_encoder
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        // Longest-first so that, at a given start position, the longest special
        // token wins even under fancy-regex's leftmost-*first* alternation.
        let mut names: Vec<&str> = special_encoder.keys().map(|s| s.as_str()).collect();
        names.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        let special_pat = if names.is_empty() {
            None
        } else {
            let alt = names
                .iter()
                .map(|s| regex_escape(s))
                .collect::<Vec<_>>()
                .join("|");
            Some(Regex::new(&alt)?)
        };

        Ok(Self {
            ranks,
            decoder,
            special_encoder,
            special_decoder,
            pat: Regex::new(pat_str)?,
            special_pat,
        })
    }

    /// Base vocab size (rank table only, no specials).
    pub fn n_base(&self) -> usize {
        self.ranks.len()
    }

    /// Total vocab size, base + special.
    pub fn n_vocab(&self) -> usize {
        self.ranks.len() + self.special_encoder.len()
    }

    /// Look up a special token's id by its literal text.
    pub fn special_id(&self, name: &str) -> Option<Rank> {
        self.special_encoder.get(name).copied()
    }

    /// Encode, reproducing the shipped `TikTokenTokenizer.encode`:
    /// the 400k-char / 25k-same-class chunking, then tiktoken's native encode
    /// per chunk. `allow_special = true` matches `allowed_special="all"`
    /// (structural markers become their ids); `false` matches
    /// `disallowed_special=()` (markup is ordinary text).
    pub fn encode(&self, text: &str, allow_special: bool) -> Vec<Rank> {
        self.encode_with(text, allow_special, byte_pair_encode, true)
    }

    /// Ordinary encode — no special-token recognition at all.
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        self.encode(text, false)
    }

    /// The one copy of the encode pipeline. `bpe` and `chunk` are knobs solely
    /// so [`mutants`] can build deliberately-wrong encoders out of the SAME
    /// code the real path runs — a mutant that shares no code with the thing it
    /// mutates proves nothing about the thing.
    fn encode_with(&self, text: &str, allow_special: bool, bpe: BpeFn, chunk: bool) -> Vec<Rank> {
        let mut out = Vec::new();
        if !chunk {
            self.encode_native(text, allow_special, &mut out, bpe);
            return out;
        }
        for chunk in char_chunks(text, TIKTOKEN_MAX_ENCODE_CHARS) {
            for sub in split_whitespaces_or_nonwhitespaces(chunk, MAX_NO_WHITESPACES_CHARS) {
                self.encode_native(sub, allow_special, &mut out, bpe);
            }
        }
        out
    }

    /// tiktoken's `_encode_native`: walk special tokens, encoding the ordinary
    /// text *between* them. Pre-tokenization never crosses a special-token
    /// boundary — the regex runs on each in-between slice, not on the whole
    /// string — which is why the two must be interleaved rather than layered.
    fn encode_native(&self, text: &str, allow_special: bool, out: &mut Vec<Rank>, bpe: BpeFn) {
        let mut start = 0usize;
        loop {
            let next_special = if allow_special {
                self.next_special(text, start)
            } else {
                None
            };
            let end = next_special.map_or(text.len(), |(s, _, _)| s);
            for m in self.pat.find_iter(&text[start..end]) {
                let piece = m.expect("pre-tokenizer regex").as_str().as_bytes();
                if let Some(&token) = self.ranks.get(piece) {
                    out.push(token);
                    continue;
                }
                out.extend(bpe(piece, &self.ranks));
            }
            match next_special {
                Some((_, e, id)) => {
                    out.push(id);
                    start = e;
                }
                None => break,
            }
        }
    }

    /// The first special token at or after `from`, as `(start, end, id)`.
    fn next_special(&self, text: &str, from: usize) -> Option<(usize, usize, Rank)> {
        let pat = self.special_pat.as_ref()?;
        let m = pat.find_from_pos(text, from).ok().flatten()?;
        let id = *self.special_encoder.get(m.as_str())?;
        Some((m.start(), m.end(), id))
    }

    /// Decode to raw bytes. Special ids decode to their literal text, as
    /// tiktoken's `decode_bytes` does. Unknown ids are skipped.
    pub fn decode_bytes(&self, ids: &[Rank]) -> Vec<u8> {
        let mut out = Vec::new();
        for id in ids {
            if let Some(b) = self.decoder.get(id) {
                out.extend_from_slice(b);
            } else if let Some(s) = self.special_decoder.get(id) {
                out.extend_from_slice(s.as_bytes());
            }
        }
        out
    }

    /// Decode to a `String`, lossily — matching tiktoken's
    /// `.decode(bytes, errors="replace")`. Byte-level BPE can split a codepoint
    /// across tokens, so a prefix of a token sequence need not be valid UTF-8.
    pub fn decode(&self, ids: &[Rank]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

/// The merge step, as a function pointer — see [`Tiktoken::encode_with`].
type BpeFn = fn(&[u8], &HashMap<Vec<u8>, Rank>) -> Vec<Rank>;

/// Merge `piece` down to tokens: repeatedly take the adjacent pair whose
/// concatenation has the LOWEST rank (leftmost wins a tie) until no adjacent
/// pair is a token. tiktoken's `_byte_pair_merge`, kept structurally identical
/// — `parts[i] = (byte offset of part i, rank of merging parts i and i+1)`,
/// with two trailing sentinels so `parts[i+3]` is always addressable.
fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    let rank_of =
        |lo: usize, hi: usize| -> Rank { ranks.get(&piece[lo..hi]).copied().unwrap_or(NO_MERGE) };

    let mut parts: Vec<(usize, Rank)> = Vec::with_capacity(piece.len() + 1);
    let mut min_rank: (Rank, usize) = (NO_MERGE, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = rank_of(i, i + 2);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, NO_MERGE));
    parts.push((piece.len(), NO_MERGE));

    // After merging at `i`, only the pair-ranks at `i-1` and `i` can change.
    let get_rank = |parts: &Vec<(usize, Rank)>, i: usize| -> Rank {
        if i + 3 < parts.len() {
            rank_of(parts[i].0, parts[i + 3].0)
        } else {
            NO_MERGE
        }
    };

    while min_rank.0 != NO_MERGE {
        let i = min_rank.1;
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (NO_MERGE, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }

    parts
        .windows(2)
        .map(|w| ranks[&piece[w[0].0..w[1].0]])
        .collect()
}

/// Escape a literal for use inside a regex alternation. Deliberately escapes
/// only the unambiguous metacharacters rather than all punctuation: fancy-regex
/// gives meaning to some backslash-punctuation pairs (`\<`, `\b`, `\K`), and the
/// special-token names are full of `<`/`>`, which are plain literals unescaped
/// but a word-boundary assertion escaped.
fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        if matches!(
            c,
            '\\' | '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Python's `str.isspace()`, which is *not* Rust's `char::is_whitespace`: it
/// additionally treats the C0 separators U+001C..U+001F as whitespace. The
/// difference only bites at a >25k-character run boundary, but the shipped
/// chunker uses `isspace`, so this does too.
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// `text[i:i+n]` in *character* units, as the shipped wrapper slices.
fn char_chunks(text: &str, n: usize) -> Vec<&str> {
    // `range(0, 0, n)` is empty, so Python yields no chunk at all for "".
    if text.is_empty() {
        return Vec::new();
    }
    if text.chars().count() <= n {
        return vec![text];
    }
    let idx: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect();
    (0..idx.len() - 1)
        .step_by(n)
        .map(|i| &text[idx[i]..idx[(i + n).min(idx.len() - 1)]])
        .collect()
}

/// Port of `TikTokenTokenizer._split_whitespaces_or_nonwhitespaces`: split so
/// no piece holds more than `max_len` consecutive same-class (space /
/// non-space) characters. Verbatim in structure, including that it always
/// yields a final piece (so the empty string yields `[""]`).
fn split_whitespaces_or_nonwhitespaces(s: &str, max_len: usize) -> Vec<&str> {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut out = Vec::new();
    let mut current_len = 0usize;
    let mut current_is_space = chars.first().map(|&(_, c)| py_isspace(c)).unwrap_or(false);
    let mut slice_start = 0usize;
    for i in 0..chars.len() {
        let is_now_space = py_isspace(chars[i].1);
        if current_is_space != is_now_space {
            current_len = 1;
            current_is_space = is_now_space;
        } else {
            current_len += 1;
            if current_len > max_len {
                out.push(&s[slice_start..chars[i].0]);
                slice_start = chars[i].0;
                current_len = 1;
            }
        }
    }
    out.push(&s[slice_start..]);
    out
}

/// Deliberately-wrong encoders, for `k3_tokenizer_gate --mutate` ONLY.
///
/// A gate that has never been seen to fail is decoration, so the ways it is
/// *supposed* to fail live in the source next to the thing they break, rather
/// than as a one-off patch someone reverted. Each of these shares the real
/// pipeline (see [`Tiktoken::encode_with`]) and changes exactly one decision.
pub mod mutants {
    use super::*;

    /// Merge the HIGHEST-rank adjacent pair first instead of the lowest — the
    /// classic direction error, and invisible on short/common words because
    /// they often have only one legal merge sequence anyway.
    pub fn encode_highest_rank_first(tk: &Tiktoken, text: &str, allow_special: bool) -> Vec<Rank> {
        tk.encode_with(text, allow_special, byte_pair_encode_worst, true)
    }

    /// Skip the shipped wrapper's 25k-same-class chunking — only observable on
    /// inputs with a >25,000-character run of whitespace or non-whitespace.
    pub fn encode_unchunked(tk: &Tiktoken, text: &str, allow_special: bool) -> Vec<Rank> {
        tk.encode_with(text, allow_special, byte_pair_encode, false)
    }

    /// [`byte_pair_encode`] with the comparison flipped: merge the highest
    /// (finite) rank first. Structure is otherwise identical.
    fn byte_pair_encode_worst(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
        if piece.len() == 1 {
            return vec![ranks[piece]];
        }
        let rank_of = |lo: usize, hi: usize| -> Rank {
            ranks.get(&piece[lo..hi]).copied().unwrap_or(NO_MERGE)
        };
        let mut parts: Vec<(usize, Rank)> = Vec::with_capacity(piece.len() + 1);
        for i in 0..piece.len() - 1 {
            parts.push((i, rank_of(i, i + 2)));
        }
        parts.push((piece.len() - 1, NO_MERGE));
        parts.push((piece.len(), NO_MERGE));
        let get_rank = |parts: &Vec<(usize, Rank)>, i: usize| -> Rank {
            if i + 3 < parts.len() {
                rank_of(parts[i].0, parts[i + 3].0)
            } else {
                NO_MERGE
            }
        };
        loop {
            let mut best: Option<(Rank, usize)> = None;
            for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
                if rank != NO_MERGE && best.map_or(true, |(r, _)| rank > r) {
                    best = Some((rank, i));
                }
            }
            let Some((_, i)) = best else { break };
            if i > 0 {
                parts[i - 1].1 = get_rank(&parts, i - 1);
            }
            parts[i].1 = get_rank(&parts, i);
            parts.remove(i + 1);
        }
        parts
            .windows(2)
            .map(|w| ranks[&piece[w[0].0..w[1].0]])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip_known_vectors() {
        // (encoded, decoded) — the first lines of a tiktoken.model, plus the
        // three padding shapes.
        assert_eq!(b64_decode(b"IQ==").unwrap(), b"!");
        assert_eq!(b64_decode(b"Ig==").unwrap(), b"\"");
        assert_eq!(b64_decode(b"IGxpYmF2dXRpbA==").unwrap(), b" libavutil");
        assert_eq!(
            b64_decode(b"6YWN5ZCI5L2/55So").unwrap(),
            "配合使用".as_bytes()
        );
        assert_eq!(b64_decode(b"").unwrap(), b"");
        assert!(b64_decode(b"IQ=").is_err()); // length not a multiple of 4
        assert!(b64_decode(b"I!==").is_err()); // outside the alphabet
    }

    #[test]
    fn parse_model_lines() {
        let src = b"IQ== 0\nIg== 1\n\nIGxpYmF2dXRpbA== 163579\n";
        let v = parse_tiktoken_model(src).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], (b"!".to_vec(), 0));
        assert_eq!(v[2], (b" libavutil".to_vec(), 163579));
    }

    #[test]
    fn merges_lowest_rank_first() {
        // "ab" ranks below "bc", so "abc" must merge to ["ab", "c"], not
        // ["a", "bc"] — the direction the whole port turns on.
        let ranks: HashMap<Vec<u8>, Rank> = [
            (b"a".to_vec(), 0),
            (b"b".to_vec(), 1),
            (b"c".to_vec(), 2),
            (b"ab".to_vec(), 3),
            (b"bc".to_vec(), 4),
        ]
        .into_iter()
        .collect();
        assert_eq!(byte_pair_encode(b"abc", &ranks), vec![3, 2]);
    }

    #[test]
    fn split_long_runs() {
        assert_eq!(split_whitespaces_or_nonwhitespaces("", 3), vec![""]);
        assert_eq!(split_whitespaces_or_nonwhitespaces("abc", 3), vec!["abc"]);
        assert_eq!(
            split_whitespaces_or_nonwhitespaces("abcd", 3),
            vec!["abc", "d"]
        );
        assert_eq!(
            split_whitespaces_or_nonwhitespaces("ab  cd", 3),
            vec!["ab  cd"]
        );
    }
}
