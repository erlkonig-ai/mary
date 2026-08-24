//! The Inkling text stack's two ends: embedding in, logits out.
//!
//! From `InklingTextModel.forward` and `InklingForCausalLM.forward`:
//!
//! ```text
//! inputs_embeds = embed_norm(embed(ids))     <- the norm runs BEFORE any layer
//! ... decoder layers ...
//! h  = final_norm(h)
//! h  = h / logits_mup_width_multiplier       <- DIVIDED, not multiplied
//! logits = unembed(h)[..., :unpadded_vocab_size]
//! ```
//!
//! Three details worth naming because each is a one-line mistake:
//!
//! * `embed_norm` is applied to the embedding output, not to the first layer's
//!   input as a layer norm would be. On the released weights it moves the
//!   embedding by ~5.9 absolute, so getting it wrong is not subtle.
//! * The muP multiplier **divides**. The name says multiplier and the code
//!   divides by it.
//! * `vocab_size` is 201024 but `unpadded_vocab_size` is 200058: the last 966
//!   columns are padding and are dropped. `embed` and `unembed` are separate
//!   tensors and are never tied.

use crate::models::inkling::block::rms_norm;

/// Look up `ids` in an embedding table stored `[vocab, hidden]`.
pub fn embed(ids: &[usize], table: &[f32], vocab: usize, hidden: usize) -> Vec<f32> {
    assert_eq!(table.len(), vocab * hidden);
    let mut out = vec![0f32; ids.len() * hidden];
    for (i, &id) in ids.iter().enumerate() {
        assert!(
            id < vocab,
            "token id {id} is past the vocabulary of {vocab}"
        );
        out[i * hidden..(i + 1) * hidden].copy_from_slice(&table[id * hidden..(id + 1) * hidden]);
    }
    out
}

/// One row of a `[vocab, hidden]` BF16 embedding table, widened.
///
/// The widening that IS allowed, and the distinction rule 3 turns on: what
/// becomes f32 here is one TOKEN's 4096 values (16 KB), not the 201024-row
/// table they came out of. Reading the table as f32 to make this a slice cost
/// 2.4 GiB of stored weight becoming 4.8 GB of host `Vec<f32>` on a box chosen
/// because the working set only just fits.
pub fn embed_row_bf16(table: &[u8], id: usize, vocab: usize, hidden: usize) -> Vec<f32> {
    assert_eq!(
        table.len(),
        vocab * hidden * 2,
        "table is not [{vocab}, {hidden}] BF16"
    );
    assert!(
        id < vocab,
        "token id {id} is past the vocabulary of {vocab}"
    );
    table[id * hidden * 2..(id + 1) * hidden * 2]
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// [`embed_and_norm`] over a table kept in the BF16 the pile stores.
///
/// Identical arithmetic: the widening of a row is exact (BF16 -> f32 loses
/// nothing), and the norm that follows is the same function over the same
/// values. What differs is that only the rows actually looked up are ever f32.
pub fn embed_and_norm_bf16(
    ids: &[usize],
    table: &[u8],
    norm_gain: &[f32],
    eps: f64,
    vocab: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut raw = vec![0f32; ids.len() * hidden];
    for (i, &id) in ids.iter().enumerate() {
        raw[i * hidden..(i + 1) * hidden]
            .copy_from_slice(&embed_row_bf16(table, id, vocab, hidden));
    }
    rms_norm(&raw, norm_gain, eps, ids.len(), hidden)
}

/// The model's input side: embedding lookup, then `embed_norm`.
pub fn embed_and_norm(
    ids: &[usize],
    table: &[f32],
    norm_gain: &[f32],
    eps: f64,
    vocab: usize,
    hidden: usize,
) -> Vec<f32> {
    let raw = embed(ids, table, vocab, hidden);
    rms_norm(&raw, norm_gain, eps, ids.len(), hidden)
}

/// The model's output side: final norm, the muP division, the head, and the
/// truncation to the unpadded vocabulary.
///
/// Returns `[tokens, unpadded_vocab]`.
pub fn head(
    hidden_states: &[f32],
    final_norm_gain: &[f32],
    unembed: &[f32],
    mup_divisor: f32,
    vocab: usize,
    unpadded_vocab: usize,
    eps: f64,
    tokens: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(hidden_states.len(), tokens * hidden);
    assert_eq!(unembed.len(), vocab * hidden);
    assert!(
        unpadded_vocab <= vocab,
        "unpadded vocabulary exceeds the table"
    );

    let normed = rms_norm(hidden_states, final_norm_gain, eps, tokens, hidden);
    let mut out = vec![0f32; tokens * unpadded_vocab];
    for t in 0..tokens {
        // Divide before the projection, matching the reference: doing it after
        // is algebraically equal and numerically not.
        let row: Vec<f32> = normed[t * hidden..(t + 1) * hidden]
            .iter()
            .map(|v| v / mup_divisor)
            .collect();
        for v in 0..unpadded_vocab {
            let w = &unembed[v * hidden..(v + 1) * hidden];
            out[t * unpadded_vocab + v] = row.iter().zip(w).map(|(a, b)| a * b).sum();
        }
    }
    out
}
