//! Qwen2.5-VL image preprocessing + prompt assembly, in pure Rust.
//!
//! Reproduces `Qwen2VLImageProcessor` semantics (the preprocessor that
//! `BiQwen2_5_Processor` / nomic-embed-multimodal-7b carries) so that an image
//! file's bytes can be turned into the exact `(pixel_values, image_grid_thw)`
//! the ported vision tower (`vision.rs`) consumes — no Python/torch in the loop.
//!
//! Pipeline (matching `image_processing_qwen2_vl.py`):
//!   1. decode + convert to RGB;
//!   2. **smart-resize** to multiples of `factor = patch_size(14) * merge(2) = 28`,
//!      keeping `min_pixels <= h*w <= max_pixels` and aspect ratio (PIL BICUBIC);
//!   3. rescale by `1/255`, normalize with the OpenAI-CLIP mean/std;
//!   4. **patchify** into `[grid_t*grid_h*grid_w, channels*temporal*patch*patch]`
//!      following the HF `view` + `permute(0,1,4,7,5,8,3,2,6,9)` + `reshape`,
//!      with the temporal-patch flattening (an image is a single frame, so
//!      `grid_t = 1` and the frame is duplicated `temporal_patch_size = 2`×).
//!
//! Prompt assembly mirrors `ColQwen2_5_Processor.visual_prompt_prefix`:
//!   `<|im_start|>user\n<|vision_start|>` + `<|image_pad|>`×(t*h*w / merge²)
//!   + `<|vision_end|>Describe the image.<|im_end|><|endoftext|>`.
//! The single `<|image_pad|>` in the template is expanded by the HF processor to
//! one placeholder per *merged* vision token; we emit that expanded count here.

use anyhow::Result;

use crate::models::gemma::gemma4::preprocess::pil_resize_bicubic;

/// `patch_size * merge_size` — both resized dims must be multiples of this.
pub const FACTOR: u32 = 28;
/// ViT patch size (px).
pub const PATCH_SIZE: usize = 14;
/// Temporal patch size: a still image is duplicated this many times.
pub const TEMPORAL_PATCH_SIZE: usize = 2;
/// Spatial merge size (2×2 patch merge in the vision tower).
pub const MERGE_SIZE: usize = 2;
/// Flattened per-patch feature width: `channels * temporal * patch * patch`.
pub const PATCH_DIM: usize = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE; // 1176

/// `min_pixels` from nomic-mm7b `preprocessor_config.json` (`56 * 56`).
pub const MIN_PIXELS: u32 = 3136;
/// `max_pixels` from nomic-mm7b `preprocessor_config.json`.
pub const MAX_PIXELS: u32 = 602112;

/// OpenAI-CLIP normalization mean (RGB).
pub const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
/// OpenAI-CLIP normalization std (RGB).
pub const IMAGE_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Python's `round()` (banker's rounding, round-half-to-even) — `smart_resize`
/// relies on it, so a naive away-from-zero round can shift the grid by a patch.
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if (diff - 0.5).abs() < 1e-9 {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        x.round()
    }
}

/// Qwen2.5-VL `smart_resize`: smallest multiple-of-`factor` `(h, w)` that keeps
/// the aspect ratio and lands the pixel count inside `[min_pixels, max_pixels]`.
/// Mirrors `image_processing_qwen2_vl.smart_resize` line-for-line.
pub fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> (u32, u32) {
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    let round_to = |v: f64| (round_half_even(v / f) * f) as u32;
    let mut h_bar = round_to(h);
    let mut w_bar = round_to(w);
    if (h_bar as u64) * (w_bar as u64) > max_pixels as u64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = factor.max(((h / beta / f).floor() as u32) * factor);
        w_bar = factor.max(((w / beta / f).floor() as u32) * factor);
    } else if ((h_bar as u64) * (w_bar as u64)) < min_pixels as u64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() as u32) * factor;
        w_bar = ((w * beta / f).ceil() as u32) * factor;
    }
    (h_bar, w_bar)
}

/// Decode image bytes and produce `(pixel_values, image_grid_thw)`:
/// - `pixel_values` is `[grid_t*grid_h*grid_w * PATCH_DIM]` row-major
///   (`[n_patches, 1176]`), the exact layout the vision tower consumes;
/// - `grid` is `(grid_t, grid_h, grid_w)` (with `grid_t = 1` for a still image).
pub fn preprocess_image(bytes: &[u8]) -> Result<(Vec<f32>, (usize, usize, usize))> {
    let img = image::load_from_memory(bytes)?.to_rgb8();
    let (w, h) = img.dimensions();
    let (rh, rw) = smart_resize(h, w, FACTOR, MIN_PIXELS, MAX_PIXELS);
    // pil_resize_bicubic is a no-op when the size is unchanged (avoid the work).
    let resized = if (rw, rh) == (w, h) {
        img
    } else {
        pil_resize_bicubic(&img, rw, rh)
    };

    let grid_t = 1usize;
    let grid_h = rh as usize / PATCH_SIZE;
    let grid_w = rw as usize / PATCH_SIZE;
    let rw_u = rw as usize;

    // Normalized pixels indexed [channel][row*rw + col]. The temporal dim simply
    // duplicates this single frame, so we don't materialize it.
    let mut norm = vec![0f32; 3 * (rh as usize) * rw_u];
    for c in 0..3 {
        let (mean, std) = (IMAGE_MEAN[c], IMAGE_STD[c]);
        let plane = c * (rh as usize) * rw_u;
        for y in 0..rh as usize {
            for x in 0..rw_u {
                let px = resized.get_pixel(x as u32, y as u32);
                let v = (px[c] as f32) / 255.0;
                norm[plane + y * rw_u + x] = (v - mean) / std;
            }
        }
    }

    // Patchify following the HF view/permute/reshape. Token (row) order:
    //   grid_t, block_row, block_col, merge_row, merge_col
    // Feature (1176) order within a row:
    //   channel, temporal, patch_row(ph), patch_col(pw)
    let n_patches = grid_t * grid_h * grid_w;
    let mut out = vec![0f32; n_patches * PATCH_DIM];
    let blocks_h = grid_h / MERGE_SIZE;
    let blocks_w = grid_w / MERGE_SIZE;
    let mut row_idx = 0usize;
    for _gt in 0..grid_t {
        for br in 0..blocks_h {
            for bc in 0..blocks_w {
                for mr in 0..MERGE_SIZE {
                    for mc in 0..MERGE_SIZE {
                        let row_base = row_idx * PATCH_DIM;
                        let mut f = 0usize;
                        for c in 0..3 {
                            let plane = c * (rh as usize) * rw_u;
                            for _tp in 0..TEMPORAL_PATCH_SIZE {
                                for ph in 0..PATCH_SIZE {
                                    let yy = br * FACTOR as usize + mr * PATCH_SIZE + ph;
                                    let yrow = plane + yy * rw_u;
                                    for pw in 0..PATCH_SIZE {
                                        let xx = bc * FACTOR as usize + mc * PATCH_SIZE + pw;
                                        out[row_base + f] = norm[yrow + xx];
                                        f += 1;
                                    }
                                }
                            }
                        }
                        debug_assert_eq!(f, PATCH_DIM);
                        row_idx += 1;
                    }
                }
            }
        }
    }
    debug_assert_eq!(row_idx, n_patches);
    Ok((out, (grid_t, grid_h, grid_w)))
}

/// Build the BiQwen2.5 image prompt string for a given vision grid. The single
/// `<|image_pad|>` of the template is expanded to one placeholder per *merged*
/// vision token: `grid_t * grid_h * grid_w / merge_size²`.
pub fn build_image_prompt(grid: (usize, usize, usize)) -> String {
    let (t, h, w) = grid;
    let n_pad = t * h * w / (MERGE_SIZE * MERGE_SIZE);
    let pads = "<|image_pad|>".repeat(n_pad);
    format!(
        "<|im_start|>user\n<|vision_start|>{pads}<|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
}
