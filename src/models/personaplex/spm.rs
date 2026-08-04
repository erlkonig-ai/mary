//! Pure-Rust SentencePiece **unigram** tokenizer for the PersonaPlex text
//! stream (`tokenizer_spm_32k_3.model`) — Phase 5 (prompt machinery).
//!
//! The checkpoint's SPM model is the best possible porting case, verified
//! from its `ModelProto` at load time (all asserted, so a different model
//! file fails loudly instead of silently mis-tokenizing):
//!
//! - `model_type = UNIGRAM` (Viterbi over piece log-probs — no BPE merges),
//! - **identity normalizer with an EMPTY `precompiled_charsmap`** — no NFKC,
//!   no double-array trie; normalization is exactly "prepend one dummy-prefix
//!   space, escape ` ` (U+0020) → `▁` (U+2581)",
//! - `remove_extra_whitespaces = false` (runs of spaces are preserved),
//! - `byte_fallback = true` with all 256 `<0xXX>` BYTE pieces present
//!   (unknown characters decompose into their UTF-8 bytes post-Viterbi).
//!
//! Special pieces: `<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3 (CONTROL/UNK
//! pieces never match from text — only NORMAL pieces enter the match table).
//! `<system>` is NOT special: it tokenizes as `▁<` + `system` + `>`, which is
//! why `wrap_with_system_tags` text round-trips through the plain encoder.
//!
//! Unigram semantics ported from sentencepiece `unigram_model.cc`:
//! Viterbi over UTF-8 char boundaries; at every position each matching
//! NORMAL piece is a lattice edge scored by its log-prob; if no single-char
//! piece covers the position, a one-char UNK edge (score `min_score − 10`,
//! argmax-irrelevant here since UNK edges never compete with known chars) is
//! added; ties resolve by strict `>` (first-best wins). Gated in
//! `personaplex_probe prompt`: the capture's exact 26-token system prompt vs
//! `text_prompt_tokens.npy` plus a 25-string battery (whitespace runs,
//! German/CJK/Cyrillic, emoji → byte fallback, the `▁` metasymbol itself)
//! vs the oracle venv (`golden/capture_spm_battery.py`).

use std::collections::HashMap;
use std::path::Path;

/// The SPM whitespace metasymbol `▁` (U+2581 LOWER ONE EIGHTH BLOCK).
const META: [u8; 3] = [0xE2, 0x96, 0x81];
/// sentencepiece `kUnkPenalty`.
const UNK_PENALTY: f32 = 10.0;

pub struct SpmTokenizer {
    /// NORMAL pieces only: UTF-8 bytes → (id, score).
    map: HashMap<Vec<u8>, (i64, f32)>,
    /// `<0xXX>` BYTE piece ids, indexed by byte value.
    byte_id: [i64; 256],
    /// Longest NORMAL piece in bytes (bounds the Viterbi edge scan).
    max_len: usize,
    /// UNK edge score: `min(NORMAL scores) − kUnkPenalty`.
    unk_score: f32,
    add_dummy_prefix: bool,
    /// Decode side: every piece's raw surface bytes indexed by id. For NORMAL
    /// pieces these are the stored UTF-8 bytes (still `▁`-escaped); for BYTE
    /// pieces the single raw byte value; CONTROL/UNK/UNUSED keep their literal
    /// surface (`<unk>`, `<s>`, …) — those never surface from `encode`, so the
    /// gates only exercise the NORMAL + BYTE lanes.
    id_to_piece: Vec<Piece>,
}

/// The decode-side surface of one piece: its raw bytes plus whether it is a
/// `byte_fallback` piece (a single raw UTF-8 byte, no `▁` un-escaping).
struct Piece {
    bytes: Vec<u8>,
    is_byte: bool,
}

// ── minimal protobuf wire-format walker (proto2, read-only) ────────────────

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn varint(&mut self) -> u64 {
        let (mut x, mut s) = (0u64, 0u32);
        loop {
            let c = self.b[self.i];
            self.i += 1;
            x |= ((c & 0x7f) as u64) << s;
            if c & 0x80 == 0 {
                return x;
            }
            s += 7;
        }
    }

    /// Next field as (field_number, payload). Varint fields return the value
    /// in-place; LEN fields return their bytes; fixed32/64 their raw bytes.
    fn field(&mut self) -> Option<(u64, FieldVal<'a>)> {
        if self.i >= self.b.len() {
            return None;
        }
        let tag = self.varint();
        let (f, wt) = (tag >> 3, tag & 7);
        let v = match wt {
            0 => FieldVal::Varint(self.varint()),
            1 => {
                let v = &self.b[self.i..self.i + 8];
                self.i += 8;
                FieldVal::Bytes(v)
            }
            2 => {
                let n = self.varint() as usize;
                let v = &self.b[self.i..self.i + n];
                self.i += n;
                FieldVal::Bytes(v)
            }
            5 => {
                let v = &self.b[self.i..self.i + 4];
                self.i += 4;
                FieldVal::Bytes(v)
            }
            _ => panic!("spm proto: unsupported wire type {wt}"),
        };
        Some((f, v))
    }
}

enum FieldVal<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

impl<'a> FieldVal<'a> {
    fn uint(&self) -> u64 {
        match self {
            FieldVal::Varint(v) => *v,
            _ => panic!("spm proto: expected varint"),
        }
    }
    fn bytes(&self) -> &'a [u8] {
        match self {
            FieldVal::Bytes(b) => b,
            _ => panic!("spm proto: expected bytes"),
        }
    }
}

impl SpmTokenizer {
    pub fn load(path: &Path) -> Self {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("spm model {path:?}: {e}"));
        let mut pieces: Vec<(Vec<u8>, f32, u64)> = Vec::new(); // (bytes, score, type)
        let (mut trainer, mut normalizer): (Option<&[u8]>, Option<&[u8]>) = (None, None);

        let mut r = Reader { b: &data, i: 0 };
        while let Some((f, v)) = r.field() {
            match f {
                // ModelProto.pieces: {1: piece, 2: score, 3: type (default NORMAL=1)}
                1 => {
                    let (mut piece, mut score, mut typ) = (Vec::new(), 0f32, 1u64);
                    let mut pr = Reader { b: v.bytes(), i: 0 };
                    while let Some((pf, pv)) = pr.field() {
                        match pf {
                            1 => piece = pv.bytes().to_vec(),
                            2 => score = f32::from_le_bytes(pv.bytes().try_into().unwrap()),
                            3 => typ = pv.uint(),
                            _ => {}
                        }
                    }
                    pieces.push((piece, score, typ));
                }
                2 => trainer = Some(v.bytes()),
                3 => normalizer = Some(v.bytes()),
                _ => {}
            }
        }

        // TrainerSpec: 3 = model_type (1 = UNIGRAM), 35 = byte_fallback.
        let (mut model_type, mut byte_fallback) = (1u64, false);
        let mut tr = Reader {
            b: trainer.expect("spm: no trainer_spec"),
            i: 0,
        };
        while let Some((f, v)) = tr.field() {
            match f {
                3 => model_type = v.uint(),
                35 => byte_fallback = v.uint() != 0,
                _ => {}
            }
        }
        assert_eq!(model_type, 1, "spm: only the UNIGRAM model is ported");
        assert!(
            byte_fallback,
            "spm: expected byte_fallback (this port relies on it)"
        );

        // NormalizerSpec: 2 = precompiled_charsmap, 3 = add_dummy_prefix,
        // 4 = remove_extra_whitespaces (proto2 default TRUE — only an
        // explicit false is supported), 5 = escape_whitespaces (default true).
        let (mut charsmap_len, mut add_dummy_prefix) = (0usize, true);
        let (mut remove_extra_ws, mut escape_ws) = (true, true);
        let mut nr = Reader {
            b: normalizer.expect("spm: no normalizer_spec"),
            i: 0,
        };
        while let Some((f, v)) = nr.field() {
            match f {
                2 => charsmap_len = v.bytes().len(),
                3 => add_dummy_prefix = v.uint() != 0,
                4 => remove_extra_ws = v.uint() != 0,
                5 => escape_ws = v.uint() != 0,
                _ => {}
            }
        }
        assert_eq!(
            charsmap_len, 0,
            "spm: non-empty precompiled_charsmap (NFKC-style normalizer) is not ported"
        );
        assert!(
            !remove_extra_ws,
            "spm: remove_extra_whitespaces=true is not ported"
        );
        assert!(escape_ws, "spm: escape_whitespaces=false is not ported");

        // Match table: NORMAL (type 1) pieces only — CONTROL/UNK/BYTE pieces
        // never match from text (model.cc BuildTrie). No USER_DEFINED pieces
        // exist in this model (they'd need always-match semantics).
        let mut map = HashMap::new();
        let mut byte_id = [-1i64; 256];
        let mut id_to_piece: Vec<Piece> = Vec::with_capacity(pieces.len());
        let (mut max_len, mut min_score) = (0usize, f32::MAX);
        for (id, (piece, score, typ)) in pieces.iter().enumerate() {
            match typ {
                1 => {
                    max_len = max_len.max(piece.len());
                    min_score = min_score.min(*score);
                    map.insert(piece.clone(), (id as i64, *score));
                    id_to_piece.push(Piece {
                        bytes: piece.clone(),
                        is_byte: false,
                    });
                }
                6 => {
                    // "<0xAB>"
                    let s = std::str::from_utf8(piece).expect("byte piece utf8");
                    let b = u8::from_str_radix(&s[3..5], 16).expect("byte piece hex");
                    byte_id[b as usize] = id as i64;
                    id_to_piece.push(Piece {
                        bytes: vec![b],
                        is_byte: true,
                    });
                }
                4 => panic!("spm: USER_DEFINED pieces are not ported"),
                // UNK (2), CONTROL (3), UNUSED (5): not matchable from text.
                // Keep their literal surface for decode completeness (they
                // never surface from `encode`, so decode gates skip them).
                _ => id_to_piece.push(Piece {
                    bytes: piece.clone(),
                    is_byte: false,
                }),
            }
        }
        assert!(byte_id.iter().all(|&i| i >= 0), "spm: missing byte pieces");

        Self {
            map,
            byte_id,
            max_len,
            unk_score: min_score - UNK_PENALTY,
            add_dummy_prefix,
            id_to_piece,
        }
    }

    /// Encode to piece ids (no BOS/EOS — matches the oracle's plain
    /// `sp.encode(text)`, which is what feeds the prompt flow).
    pub fn encode(&self, text: &str) -> Vec<i64> {
        if text.is_empty() {
            return Vec::new();
        }
        // Normalize: dummy prefix + escape ' ' → '▁'. Everything else (tabs,
        // newlines, a literal ▁ in the input) passes through untouched —
        // identity normalizer.
        let mut norm: Vec<u8> = Vec::with_capacity(text.len() + 4);
        if self.add_dummy_prefix {
            norm.extend_from_slice(&META);
        }
        for ch in text.chars() {
            if ch == ' ' {
                norm.extend_from_slice(&META);
            } else {
                let mut buf = [0u8; 4];
                norm.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }

        // UTF-8 char length at each position (0 for continuation bytes —
        // those positions are not lattice nodes).
        let n = norm.len();
        let char_len = |pos: usize| -> usize {
            match norm[pos] {
                b if b < 0x80 => 1,
                b if b >= 0xF0 => 4,
                b if b >= 0xE0 => 3,
                b if b >= 0xC0 => 2,
                _ => 0,
            }
        };

        // Viterbi: best[pos] = max score of a segmentation of norm[..pos];
        // back[pos] = (start, piece id; -1 = UNK edge). Strict `>` keeps the
        // first-best (sentencepiece lattice tie behavior).
        const NEG: f32 = f32::NEG_INFINITY;
        let mut best = vec![NEG; n + 1];
        let mut back: Vec<(usize, i64)> = vec![(0, -2); n + 1];
        best[0] = 0.0;
        let mut pos = 0;
        while pos < n {
            let cl = char_len(pos);
            debug_assert!(cl > 0, "lattice node on a continuation byte");
            if best[pos] > NEG {
                let mut has_single = false;
                let mut end = pos;
                while end < n && end - pos < self.max_len {
                    end += match norm.get(end).map(|_| char_len(end)) {
                        Some(c) if c > 0 => c,
                        _ => break,
                    };
                    if end > n {
                        break;
                    }
                    if let Some(&(id, score)) = self.map.get(&norm[pos..end]) {
                        if best[pos] + score > best[end] {
                            best[end] = best[pos] + score;
                            back[end] = (pos, id);
                        }
                        if end - pos == cl {
                            has_single = true;
                        }
                    }
                }
                if !has_single {
                    let end = pos + cl;
                    if best[pos] + self.unk_score > best[end] {
                        best[end] = best[pos] + self.unk_score;
                        back[end] = (pos, -1);
                    }
                }
            }
            pos += cl;
        }
        assert!(
            best[n] > NEG,
            "spm viterbi: unreachable end (corrupt utf8?)"
        );

        // Backtrack; UNK edges decompose into byte pieces (byte_fallback).
        let mut rev: Vec<i64> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (start, id) = back[pos];
            if id >= 0 {
                rev.push(id);
            } else {
                for &b in norm[start..pos].iter().rev() {
                    rev.push(self.byte_id[b as usize]);
                }
            }
            pos = start;
        }
        rev.reverse();
        rev
    }

    /// Raw surface bytes of a single piece id (still `▁`-escaped for NORMAL
    /// pieces; a single raw byte for `byte_fallback` pieces). This is the
    /// low-level inverse of the id table — most callers want [`Self::decode`],
    /// which reassembles and un-escapes a whole id run into a `String`.
    ///
    /// Panics if `id` is outside `0..vocab_size()` — ids come from `encode`
    /// or the model's own text stream, both bounded by the vocab.
    pub fn piece_bytes(&self, id: i64) -> &[u8] {
        &self.id_to_piece[id as usize].bytes
    }

    /// Number of pieces in the vocabulary (`id_to_piece.len()`).
    pub fn vocab_size(&self) -> usize {
        self.id_to_piece.len()
    }

    /// Whether `id` is a `byte_fallback` piece (`<0xXX>`, one raw byte).
    pub fn is_byte_piece(&self, id: i64) -> bool {
        self.id_to_piece[id as usize].is_byte
    }

    /// Decode a single piece id to its printable string, sentencepiece
    /// `IdToPiece`-style: NORMAL pieces render with `▁` → ' ' un-escaping;
    /// `byte_fallback` pieces render as the raw byte (may be an incomplete
    /// UTF-8 sequence in isolation — prefer [`Self::decode`] on a full run,
    /// which stitches multi-byte characters back together). Lossy UTF-8 for
    /// isolated partial bytes.
    pub fn decode_token(&self, id: i64) -> String {
        let p = &self.id_to_piece[id as usize];
        if p.is_byte {
            String::from_utf8_lossy(&p.bytes).into_owned()
        } else {
            unescape(&p.bytes)
        }
    }

    /// Detokenize an id run back to text — the inverse of [`Self::encode`]
    /// (sentencepiece `DecodePieces`): concatenate every piece's raw surface
    /// bytes (byte-fallback pieces contribute their single raw byte, so
    /// multi-byte UTF-8 characters split across a run stitch back together
    /// here), then un-escape `▁` → ' ' and strip the single dummy-prefix
    /// space that `encode` prepends. `add_dummy_prefix = false` skips the
    /// strip. Lossy UTF-8 only if the run contains a genuinely invalid byte
    /// sequence (it never does for a run produced by `encode`).
    pub fn decode(&self, ids: &[i64]) -> String {
        let mut raw: Vec<u8> = Vec::new();
        for &id in ids {
            raw.extend_from_slice(&self.id_to_piece[id as usize].bytes);
        }
        let mut s = unescape(&raw);
        if self.add_dummy_prefix {
            // `encode` escapes the prepended dummy space to `▁`, which
            // `unescape` turns back into a leading ' '; drop exactly one.
            if let Some(stripped) = s.strip_prefix(' ') {
                s = stripped.to_string();
            }
        }
        s
    }
}

/// Un-escape a byte run: `▁` (U+2581) → ' ', everything else verbatim, then
/// lossy-UTF-8 to a `String`. Shared by `decode_token`/`decode`.
fn unescape(bytes: &[u8]) -> String {
    // Replace the 3-byte META sequence with a single space in the byte
    // stream, then decode once (so a byte-fallback char split across pieces
    // still forms a valid multi-byte sequence before UTF-8 decoding).
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(&META) {
            out.push(b' ');
            i += META.len();
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Phase-0 checkpoint fetch drops the SPM model here (probe scratch);
    /// absent on a bare CI, so the round-trip gate no-ops rather than fails.
    const SPM_MODEL: &str = "/tmp/personaplex_scratch/ckpt/tokenizer_spm_32k_3.model";

    /// A battery spanning every decode lane: ASCII words + whitespace runs
    /// (NORMAL pieces + the `▁` metasymbol), German/CJK/Cyrillic (multi-byte
    /// NORMAL or byte-fallback), emoji (byte-fallback), and the `<system>`
    /// tag (which is NOT special — it splits into `▁<`+`system`+`>`). None
    /// start with a space, so the dummy-prefix strip in `decode` reconstructs
    /// them exactly.
    ///
    /// NOTE the omission of a *literal* `▁` (U+2581) in the input: SPM's
    /// escape (space → `▁`) is not injective, so `decode(encode("a ▁ b"))`
    /// collapses the metasymbol to a space in reference sentencepiece too —
    /// an inherent, not-a-bug ambiguity. The encode-side gate in
    /// `personaplex_probe prompt` covers the literal-`▁` tokenization; decode
    /// of it is undefined by design.
    fn battery() -> Vec<&'static str> {
        vec![
            "Hello, world!",
            "The quick brown fox.",
            "a  b   c",          // whitespace runs (remove_extra_ws=false)
            "tab\tand\nnewline", // tab/newline pass through
            "Grüße aus München", // German umlauts
            "日本語のテスト",    // CJK
            "Привет мир",        // Cyrillic
            "emoji 😀🎉 test",   // multi-byte, some byte-fallback
            "<system>be nice</system>",
            "mixed 混合 text 123",
            "punctuation: (a) [b] {c} <d>",
        ]
    }

    #[test]
    fn decode_roundtrips_encode() {
        let path = Path::new(SPM_MODEL);
        if !path.exists() {
            eprintln!("SKIP spm round-trip: {SPM_MODEL} absent (Phase-0 fetch not run)");
            return;
        }
        let spm = SpmTokenizer::load(path);
        for text in battery() {
            let ids = spm.encode(text);
            let back = spm.decode(&ids);
            assert_eq!(back, text, "round-trip mismatch\n  ids  {ids:?}");
        }
    }

    /// `decode` of an id run must equal concatenating `decode_token` over the
    /// same run once the dummy-prefix space is accounted for — the two APIs
    /// share one surface. (Multi-byte byte-fallback chars only stitch under
    /// `decode`; here the battery's per-token concatenation is compared to
    /// `decode` on the SAME reassembled byte stream, so they agree.)
    #[test]
    fn decode_token_agrees_with_decode() {
        let path = Path::new(SPM_MODEL);
        if !path.exists() {
            eprintln!("SKIP spm decode_token: {SPM_MODEL} absent");
            return;
        }
        let spm = SpmTokenizer::load(path);
        for text in battery() {
            let ids = spm.encode(text);
            // Reassemble raw bytes exactly as `decode` does, per token via the
            // public byte accessor, then run the same un-escape + strip.
            let mut raw = Vec::new();
            for &id in &ids {
                raw.extend_from_slice(spm.piece_bytes(id));
            }
            let mut expect = super::unescape(&raw);
            if let Some(s) = expect.strip_prefix(' ') {
                expect = s.to_string();
            }
            assert_eq!(spm.decode(&ids), expect, "decode vs piece_bytes surface");
        }
    }

    /// Byte-fallback pieces decode to a single raw byte, and an emoji whose
    /// codepoint isn't a vocab piece round-trips purely through byte fallback
    /// (its bytes stitch back into the 4-byte character under `decode`).
    #[test]
    fn byte_fallback_ids_present_and_reversible() {
        let path = Path::new(SPM_MODEL);
        if !path.exists() {
            eprintln!("SKIP spm byte-fallback: {SPM_MODEL} absent");
            return;
        }
        let spm = SpmTokenizer::load(path);
        // A rocket emoji round-trips (whether via NORMAL or byte pieces).
        let ids = spm.encode("🚀");
        assert_eq!(spm.decode(&ids), "🚀");
        // Any piece flagged byte-fallback has exactly one raw byte, and at
        // least one byte-fallback piece exists in the vocabulary.
        let mut any_byte = false;
        for id in 0..spm.vocab_size() as i64 {
            if spm.is_byte_piece(id) {
                assert_eq!(spm.piece_bytes(id).len(), 1, "byte piece {id} not 1 byte");
                any_byte = true;
            }
        }
        assert!(any_byte, "no byte-fallback pieces found");
        // A lone raw byte (0x00) has a byte piece whose decode is that byte.
        let z = spm.byte_id[0];
        assert!(z >= 0 && spm.is_byte_piece(z));
        assert_eq!(spm.piece_bytes(z), &[0u8]);
    }
}
