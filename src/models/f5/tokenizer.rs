//! F5 char/pinyin tokenizer. For pure-ASCII (English) text, F5's
//! `convert_char_to_pinyin` + `list_str_to_idx` reduces to a plain char→id
//! lookup (verified identical on the demo text), so no rjieba/pinyin is needed.
//! The vocab is embedded (`assets/vocab.txt`) so the binary is self-contained.

use burn::prelude::*;
use std::collections::HashMap;

const VOCAB: &str = include_str!("../../../assets/vocab.txt");

pub struct Tokenizer {
    map: HashMap<String, i64>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        for (i, line) in VOCAB.lines().enumerate() {
            map.insert(line.to_string(), i as i64);
        }
        Self { map }
    }

    /// Map each character to its vocab id; space and unknown → 0 (filler).
    pub fn encode(&self, text: &str) -> Vec<i64> {
        text.chars()
            .map(|c| *self.map.get(&c.to_string()).unwrap_or(&0))
            .collect()
    }

    /// Encode to a [1, n_chars] Int tensor of raw ids (pre-`+1`).
    pub fn encode_tensor<B: Backend>(&self, text: &str, device: &B::Device) -> Tensor<B, 2, Int> {
        let ids: Vec<i32> = self.encode(text).into_iter().map(|i| i as i32).collect();
        let n = ids.len();
        Tensor::<B, 1, Int>::from_data(burn::tensor::TensorData::new(ids, [n]), device)
            .reshape([1, n])
    }
}
