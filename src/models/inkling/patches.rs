//! The patch front end for Inkling's eye: an RGB8 image in, its patches out.
//!
//! This is a TOKENIZER, not model math, exactly like [`super::dmel`] for the
//! ear: a cut into 40-pixel squares, a pad, a rescale and a normalisation,
//! with no learned parameter in it. It runs where the pixels are. The model's
//! side -- the four-layer HMLP that folds a patch down to one text-width row
//! and norms it -- is the Session's, on the device.
//!
//! The steps are read off the shipped `InklingImageProcessor`
//! (`image_processing_inkling.py`, `processor_config.json`) rather than
//! remembered, because three of them are the kind that get remembered wrong:
//!
//! - the grid is `ceil(height / 40)` rows by `width / 40 + 1` columns -- one
//!   MORE column than the width fills, always, even when the width is a
//!   multiple of forty (the reference's `divide_to_patches` is "slightly
//!   different" from the library one, and this is the difference);
//! - the pad fill is `-1.0` on the RAW 0..255 scale, applied BEFORE the
//!   rescale and normalisation, so a padded value is `(-1/255 - mean) / std`,
//!   not zero and not `-1`;
//! - a still image is stacked twice along time (`temporal_patch_size` 2) and
//!   the patch is laid out `[t][y][x][c]`, which is what the reference's final
//!   `permute(0, 4, 2, 3, 1)` produces.
//!
//! What it deliberately does not do: resize. The eye faculty owns the long
//! edge it sends (the reference's optional `rescale_image_frac`), because the
//! camera's resolution and the token budget are its business.

pub const PATCH: usize = 40;
pub const TEMPORAL: usize = 2;
pub const CHANNELS: usize = 3;
/// Values in one patch: `2 * 40 * 40 * 3`.
pub const PATCH_VALUES: usize = TEMPORAL * PATCH * PATCH * CHANNELS;
/// Bytes of one patch on the wire: little-endian f32s.
pub const PATCH_BYTES: usize = PATCH_VALUES * 4;

const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
const RESCALE: f32 = 1.0 / 255.0;
const PAD_FILL: f32 = -1.0;

/// `(rows, cols)` of the patch grid over a `width x height` image.
pub fn patch_grid(width: usize, height: usize) -> (usize, usize) {
    (height.div_ceil(PATCH), width / PATCH + 1)
}

/// How many patches a `width x height` image becomes.
pub fn patch_count(width: usize, height: usize) -> usize {
    let (rows, cols) = patch_grid(width, height);
    rows * cols
}

/// The normalised value of one channel's pixel, or of the pad where there is
/// no pixel.
fn norm(raw: f32, c: usize) -> f32 {
    (raw * RESCALE - MEAN[c]) / STD[c]
}

/// A still image's patches: `patch_count(width, height) * PATCH_VALUES` f32s,
/// patch-major in row-major grid order, each `[t][y][x][c]` with both time
/// steps the same frame. `rgb` is `width * height * 3` bytes, row-major.
pub fn still(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    video(rgb, rgb, width, height)
}

/// Two consecutive frames' patches, `t = 0` from `first` and `t = 1` from
/// `second`; the shape is otherwise [`still`]'s.
pub fn video(first: &[u8], second: &[u8], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(first.len(), width * height * CHANNELS, "first frame is not {width}x{height} RGB8");
    assert_eq!(second.len(), width * height * CHANNELS, "second frame is not {width}x{height} RGB8");
    let (rows, cols) = patch_grid(width, height);
    let mut out = Vec::with_capacity(rows * cols * PATCH_VALUES);
    for i in 0..rows {
        for j in 0..cols {
            for frame in [first, second] {
                for y in 0..PATCH {
                    let py = i * PATCH + y;
                    for x in 0..PATCH {
                        let px = j * PATCH + x;
                        if py < height && px < width {
                            let at = (py * width + px) * CHANNELS;
                            for c in 0..CHANNELS {
                                out.push(norm(f32::from(frame[at + c]), c));
                            }
                        } else {
                            for c in 0..CHANNELS {
                                out.push(norm(PAD_FILL, c));
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// The wire form: little-endian f32s, [`PATCH_BYTES`] per patch.
pub fn to_bytes(patches: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(patches.len() * 4);
    for v in patches {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Back from the wire; refuses anything that is not whole patches.
pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len() % PATCH_BYTES == 0,
        "a patch record is {PATCH_BYTES} bytes per patch; {} arrived",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// How many patches `bytes` holds, with the same refusal as [`from_bytes`].
pub fn count(bytes: &[u8]) -> anyhow::Result<usize> {
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len() % PATCH_BYTES == 0,
        "a patch record is {PATCH_BYTES} bytes per patch; {} arrived",
        bytes.len()
    );
    Ok(bytes.len() / PATCH_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shape checks only, by the standing rule: the grid has its extra
    /// column, the pad is the pad, a still repeats itself in time, and the
    /// wire round-trips.
    #[test]
    fn the_grid_pads_and_repeats_as_the_reference_does() {
        // Exactly one patch of pixels still makes two patches: the extra column.
        assert_eq!(patch_grid(40, 40), (1, 2));
        assert_eq!(patch_grid(100, 50), (2, 3));
        assert_eq!(patch_count(320, 180), 5 * 9);

        let (w, h) = (100usize, 50usize);
        let mut rgb = vec![0u8; w * h * 3];
        for (i, v) in rgb.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let p = still(&rgb, w, h);
        assert_eq!(p.len(), 6 * PATCH_VALUES);
        // A still: t = 0 equals t = 1 in every patch.
        let half = PATCH * PATCH * CHANNELS;
        for patch in p.chunks_exact(PATCH_VALUES) {
            assert_eq!(&patch[..half], &patch[half..]);
        }
        // Patch (0, 0) pixel (0, 0) is the image's first pixel, normalised.
        for c in 0..3 {
            assert_eq!(p[c], norm(f32::from(rgb[c]), c));
        }
        // The last column is entirely pad (x = 80.. is past width 100? no:
        // cols = 100/40 + 1 = 3, so column 2 covers x 80..120, half pad).
        let last = &p[2 * PATCH_VALUES..3 * PATCH_VALUES];
        let at = |y: usize, x: usize, c: usize| last[(y * PATCH + x) * CHANNELS + c];
        assert_eq!(at(0, 19, 0), norm(f32::from(rgb[(0 * w + 99) * 3]), 0));
        assert_eq!(at(0, 20, 0), norm(PAD_FILL, 0));
        // Row 1 covers y 40..80 over height 50: y = 10.. is pad.
        let row1 = &p[3 * PATCH_VALUES..4 * PATCH_VALUES];
        assert_eq!(row1[(10 * PATCH) * CHANNELS + 1], norm(PAD_FILL, 1));
        assert_ne!(row1[(9 * PATCH) * CHANNELS + 1], norm(PAD_FILL, 1));

        let bytes = to_bytes(&p);
        assert_eq!(bytes.len(), 6 * PATCH_BYTES);
        assert_eq!(count(&bytes).unwrap(), 6);
        assert_eq!(from_bytes(&bytes).unwrap(), p);
        assert!(from_bytes(&bytes[..PATCH_BYTES - 4]).is_err());
        assert!(count(&[]).is_err());
    }
}
