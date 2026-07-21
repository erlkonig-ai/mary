//! Tekken tokenizer, DECODE-ONLY — ASR emits ids, we only need id → bytes.
//! `tekken.json` layout: ids `0..1000` are special tokens (skipped in
//! transcripts), id `i >= 1000` maps to `vocab[i − 1000].token_bytes`
//! (base64-encoded raw bytes; concatenate, then UTF-8).

use std::path::Path;

/// Minimal base64 (standard alphabet, `=` padding) — avoids a dep for one call.
fn b64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> i32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as i32,
            b'a'..=b'z' => (c - b'a') as i32 + 26,
            b'0'..=b'9' => (c - b'0') as i32 + 52,
            b'+' => 62,
            b'/' => 63,
            _ => -1,
        }
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in s.as_bytes() {
        let v = val(c);
        if v < 0 {
            continue; // '=' padding / whitespace
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

pub struct Tekken {
    /// `vocab[i]` = raw bytes of model token id `i + 1000`.
    vocab: Vec<Vec<u8>>,
    pub num_special: u32,
}

impl Tekken {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&raw)?;
        let num_special = json["config"]["default_num_special_tokens"]
            .as_u64()
            .unwrap_or(1000) as u32;
        let vocab = json["vocab"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tekken.json: no vocab array"))?
            .iter()
            .map(|e| {
                e["token_bytes"]
                    .as_str()
                    .map(b64_decode)
                    .ok_or_else(|| anyhow::anyhow!("tekken.json: vocab entry without token_bytes"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { vocab, num_special })
    }

    /// Decode a token stream to text, skipping special ids (BOS/EOS/pads).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id < self.num_special {
                continue;
            }
            if let Some(b) = self.vocab.get((id - self.num_special) as usize) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
