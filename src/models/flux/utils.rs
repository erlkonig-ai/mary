use burn::prelude::*;
use burn::tensor::TensorData;

/// Pack latents: [B, C, H, W] -> [B, H*W, C]
pub fn pack_latents<B: Backend>(latents: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, channels, height, width] = latents.dims();
    latents
        .reshape([batch, channels, height * width])
        .swap_dims(1, 2) // [B, H*W, C]
}

/// Unpack latents: [B, H*W, C] -> [B, C, H, W]
/// Uses position IDs to scatter tokens back to spatial positions.
pub fn unpack_latents_with_ids<B: Backend>(
    x: Tensor<B, 3>,     // [B, seq, C]
    x_ids: Tensor<B, 3>, // [B, seq, 4] — position IDs (t, h, w, l)
    device: &B::Device,
) -> Tensor<B, 4> {
    let [batch, _seq, channels] = x.dims();

    // Extract h and w coordinates from position IDs
    // Pull all the ID data at once to avoid tricky squeeze semantics
    let ids_data: Vec<f32> = x_ids.to_data().to_vec().unwrap();

    // Build h_ids and w_ids vectors from the raw data
    // ids_data layout: [B, seq, 4] flattened — indices: (b * seq * 4) + (s * 4) + coord
    let mut h_ids_data = Vec::with_capacity(batch * _seq);
    let mut w_ids_data = Vec::with_capacity(batch * _seq);
    for b in 0..batch {
        for s in 0.._seq {
            let base = (b * _seq + s) * 4;
            h_ids_data.push(ids_data[base + 1]); // h coord
            w_ids_data.push(ids_data[base + 2]); // w coord
        }
    }

    let height = h_ids_data.iter().cloned().fold(0.0_f32, f32::max) as usize + 1;
    let width = w_ids_data.iter().cloned().fold(0.0_f32, f32::max) as usize + 1;

    // Scatter tokens into spatial positions
    // For each batch element, create output [C, H, W]
    let x_data: Vec<f32> = x.to_data().to_vec().unwrap();

    let mut out = vec![0.0f32; batch * channels * height * width];

    for b in 0..batch {
        for s in 0.._seq {
            let h = h_ids_data[b * _seq + s] as usize;
            let w = w_ids_data[b * _seq + s] as usize;
            for c in 0..channels {
                let src_idx = b * _seq * channels + s * channels + c;
                let dst_idx = b * channels * height * width + c * height * width + h * width + w;
                out[dst_idx] = x_data[src_idx];
            }
        }
    }

    Tensor::from_data(
        TensorData::new(out, [batch, channels, height, width]),
        device,
    )
}

/// Patchify latents: [B, C, H, W] -> [B, C*4, H/2, W/2]
/// Groups 2x2 spatial patches into the channel dimension.
pub fn patchify_latents<B: Backend>(latents: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = latents.dims();
    assert!(height % 2 == 0 && width % 2 == 0, "H and W must be even");
    // [B, C, H, W] -> [B, C, H/2, 2, W/2, 2]
    let reshaped = latents.reshape([batch, channels, height / 2, 2, width / 2, 2]);
    // permute to [B, C, 2, 2, H/2, W/2]
    let permuted = reshaped.permute([0, 1, 3, 5, 2, 4]);
    // reshape to [B, C*4, H/2, W/2]
    permuted.reshape([batch, channels * 4, height / 2, width / 2])
}

/// Unpatchify latents: [B, C*4, H/2, W/2] -> [B, C, H, W]
/// Inverse of patchify.
pub fn unpatchify_latents<B: Backend>(latents: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, num_channels, height, width] = latents.dims();
    let channels = num_channels / 4;
    // [B, C*4, H/2, W/2] -> [B, C, 2, 2, H/2, W/2]
    let reshaped = latents.reshape([batch, channels, 2, 2, height, width]);
    // permute to [B, C, H/2, 2, W/2, 2]
    let permuted = reshaped.permute([0, 1, 4, 2, 5, 3]);
    // reshape to [B, C, H, W]
    permuted.reshape([batch, channels, height * 2, width * 2])
}

/// Prepare text position IDs: [B, L, 4] with (t=0, h=0, w=0, l=0..L-1)
pub fn prepare_text_ids<B: Backend>(batch_size: usize, seq_len: usize, device: &B::Device) -> Tensor<B, 3> {
    // For each token position l, the ID is (0, 0, 0, l)
    let mut data = vec![0.0f32; batch_size * seq_len * 4];
    for b in 0..batch_size {
        for l in 0..seq_len {
            let base = (b * seq_len + l) * 4;
            data[base] = 0.0;     // t
            data[base + 1] = 0.0; // h
            data[base + 2] = 0.0; // w
            data[base + 3] = l as f32; // l
        }
    }
    Tensor::from_data(
        TensorData::new(data, [batch_size, seq_len, 4]),
        device,
    )
}

/// Prepare latent (image) position IDs: [B, H*W, 4] with (t=0, h=0..H-1, w=0..W-1, l=0)
pub fn prepare_latent_ids<B: Backend>(
    batch_size: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let seq_len = height * width;
    let mut data = vec![0.0f32; batch_size * seq_len * 4];
    for b in 0..batch_size {
        for h in 0..height {
            for w in 0..width {
                let idx = b * seq_len + h * width + w;
                let base = idx * 4;
                data[base] = 0.0;     // t
                data[base + 1] = h as f32; // h
                data[base + 2] = w as f32; // w
                data[base + 3] = 0.0; // l
            }
        }
    }
    Tensor::from_data(
        TensorData::new(data, [batch_size, seq_len, 4]),
        device,
    )
}

/// Convert output tensor [B, 3, H, W] (values in [-1, 1]) to an RGB image.
// The 0/1/2 × height·width channel offsets are deliberate parallel structure
// (clippy's erasing_op deny would force `data[y*width + x]` for red only).
#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn tensor_to_image<B: Backend>(tensor: Tensor<B, 4>) -> image::RgbImage {
    let [_b, _c, height, width] = tensor.dims();
    // Take first batch element
    let tensor: Tensor<B, 3> = tensor.slice([0..1]).squeeze(); // [3, H, W]

    // Scale from [-1, 1] to [0, 255]
    let tensor = (tensor + 1.0) * 127.5;
    let tensor = tensor.clamp(0.0, 255.0);

    let data: Vec<f32> = tensor.to_data().to_vec().unwrap();

    let mut img = image::RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let r = data[0 * height * width + y * width + x] as u8;
            let g = data[1 * height * width + y * width + x] as u8;
            let b_val = data[2 * height * width + y * width + x] as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b_val]));
        }
    }
    img
}
