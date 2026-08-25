//! Load Ministral weights from HuggingFace safetensors format or TribleSpace pile.
//!
//! Safetensors: prefix stripping, bf16→f32, Linear transpose, weight tying.
//! Pile: per-tensor F32Array blobs with TribleSpace metadata entities.

use burn::prelude::*;
#[cfg(feature = "import")]
use memmap2::Mmap;
#[cfg(feature = "import")]
use safetensors::SafeTensors;
#[cfg(feature = "import")]
use std::collections::HashMap;
#[cfg(feature = "import")]
use std::path::Path;

#[cfg(feature = "import")]
use crate::models::gemma::config::MistralConfig;
#[cfg(feature = "import")]
use crate::models::gemma::decoder::MistralModel;

/// Convert raw bytes to f32 based on the safetensors dtype.
/// Convert raw bytes to f32 based on the safetensors dtype.
#[cfg(feature = "import")]
pub fn bytes_to_f32_pub(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    bytes_to_f32(data, dtype)
}

#[cfg(feature = "import")]
fn bytes_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect(),
        safetensors::Dtype::F16 => data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        safetensors::Dtype::F8_E4M3 => {
            // FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
            // Range: [-448, 448], needs scale_inv for dequantization
            // Each byte is one FP8 value. Convert to f32 via the format spec.
            data.iter().map(|&byte| fp8_e4m3_to_f32(byte)).collect()
        }
        other => panic!("Unsupported dtype: {:?}", other),
    }
}

/// Convert a single FP8 E4M3FN byte to f32.
#[cfg(feature = "import")]
fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF; // 4 bits
    let mant = bits & 0x7; // 3 bits

    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }

    let value = if exp == 0 {
        // Subnormal: 2^(-6) * (mant / 8)
        (mant as f32 / 8.0) * 2.0f32.powi(-6)
    } else {
        // Normal: 2^(exp-7) * (1 + mant/8)
        (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7)
    };

    if sign == 1 { -value } else { value }
}

/// Load a tensor from safetensors, converting to f32 and optionally transposing.
/// Handles BF16, F16, F32, and FP8 (with scale_inv dequantization).
#[cfg(feature = "import")]
fn load_tensor<B: Backend>(
    tensors: &HashMap<String, &SafeTensors<'_>>,
    name: &str,
    device: &B::Device,
    transpose: bool,
) -> Tensor<B, 2> {
    let (st, local_name) = find_tensor(tensors, name);
    let view = st
        .tensor(local_name)
        .unwrap_or_else(|e| panic!("Missing tensor {name}: {e}"));
    let data = view.data();
    let shape = view.shape();
    let dtype = view.dtype();

    let mut f32_data = bytes_to_f32(data, dtype);

    // For FP8, apply scale_inv dequantization if available
    if matches!(dtype, safetensors::Dtype::F8_E4M3) {
        let scale_name = format!(
            "{}.weight_scale_inv",
            name.strip_suffix(".weight").unwrap_or(name)
        );
        if let Ok((scale_st, scale_local)) =
            std::panic::catch_unwind(|| find_tensor(tensors, &scale_name))
        {
            let scale_view = scale_st.tensor(scale_local).unwrap();
            let scale_bytes = scale_view.data();
            let scale = if scale_view.dtype() == safetensors::Dtype::BF16 {
                let bits = u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]);
                half::bf16::from_bits(bits).to_f32()
            } else {
                f32::from_le_bytes([
                    scale_bytes[0],
                    scale_bytes[1],
                    scale_bytes[2],
                    scale_bytes[3],
                ])
            };
            for v in &mut f32_data {
                *v *= scale;
            }
        }
    }

    let rows = shape[0];
    let cols = shape[1];
    let tensor = Tensor::<B, 1>::from_floats(&f32_data[..], device).reshape([rows, cols]);

    if transpose {
        tensor.swap_dims(0, 1)
    } else {
        tensor
    }
}

/// Load a 1D tensor (for norms).
#[cfg(feature = "import")]
fn load_tensor_1d<B: Backend>(
    tensors: &HashMap<String, &SafeTensors<'_>>,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 1> {
    let (st, local_name) = find_tensor(tensors, name);
    let view = st
        .tensor(local_name)
        .unwrap_or_else(|e| panic!("Missing tensor {name}: {e}"));
    let f32_data = bytes_to_f32(view.data(), view.dtype());
    Tensor::<B, 1>::from_floats(&f32_data[..], device)
}

/// Find which safetensors shard contains a given tensor name.
/// Searches with multiple prefix strategies:
/// 1. Exact name
/// 2. "language_model.model." prefix (most layer weights)
/// 3. "language_model." prefix (lm_head in multimodal models)
#[cfg(feature = "import")]
fn find_tensor<'a, 'b>(
    tensors: &'a HashMap<String, &'b SafeTensors<'b>>,
    name: &str,
) -> (&'b SafeTensors<'b>, &'a str) {
    // Try exact name first
    for (_shard_name, st) in tensors {
        if st.tensor(name).is_ok() {
            return (st, leak_str(name));
        }
    }
    // Try with "model." prefix (Qwen3 / standard HF format)
    let prefixed_model = format!("model.{}", name);
    for (_shard_name, st) in tensors {
        if st.tensor(&prefixed_model).is_ok() {
            return (st, leak_str(&prefixed_model));
        }
    }
    // Try with "language_model.model." prefix (Mistral multimodal)
    let prefixed = format!("language_model.model.{}", name);
    for (_shard_name, st) in tensors {
        if st.tensor(&prefixed).is_ok() {
            return (st, leak_str(&prefixed));
        }
    }
    // Try with "language_model." prefix (lm_head in multimodal models)
    let prefixed2 = format!("language_model.{}", name);
    for (_shard_name, st) in tensors {
        if st.tensor(&prefixed2).is_ok() {
            return (st, leak_str(&prefixed2));
        }
    }
    panic!(
        "Tensor not found in any shard: {name} (tried model.{name}, language_model.model.{name}, language_model.{name})"
    );
}

/// Leak a string to get a &'static str. Used for tensor name lookups.
/// This is a small memory leak but acceptable for model loading (happens once).
#[cfg(feature = "import")]
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// Create an RmsNorm module with a loaded gamma weight tensor.
pub fn make_rms_norm<B: Backend>(
    weight: Tensor<B, 1>,
    eps: f64,
    device: &B::Device,
) -> burn::nn::RmsNorm<B> {
    let [dim] = weight.dims();
    let norm = burn::nn::RmsNormConfig::new(dim)
        .with_epsilon(eps)
        .init(device);
    let record = burn::nn::RmsNormRecord {
        gamma: burn::module::Param::from_tensor(weight),
        epsilon: burn::module::ConstantRecord::new(),
    };
    norm.load_record(record)
}

/// Set a Linear module's weight from a loaded tensor.
pub fn set_linear_weight<B: Backend>(linear: &mut burn::nn::Linear<B>, weight: Tensor<B, 2>) {
    // burn's Linear stores weight as a Param. We need to reconstruct.
    // For now, use the record mechanism.
    let record = burn::nn::LinearRecord {
        weight: burn::module::Param::from_tensor(weight),
        bias: None,
    };
    *linear = linear.clone().load_record(record);
}

/// Load Ministral weights from safetensors files.
#[cfg(feature = "import")]
pub fn load_ministral<B: Backend>(
    config: MistralConfig,
    paths: &[&Path],
    device: &B::Device,
) -> MistralModel<B> {
    // Memory-map all shard files
    let files: Vec<_> = paths
        .iter()
        .map(|p| {
            let file = std::fs::File::open(p).unwrap_or_else(|e| panic!("Can't open {:?}: {e}", p));
            unsafe { Mmap::map(&file) }.unwrap_or_else(|e| panic!("Can't mmap {:?}: {e}", p))
        })
        .collect();

    let safetensors: Vec<_> = files
        .iter()
        .map(|mmap| {
            SafeTensors::deserialize(mmap)
                .unwrap_or_else(|e| panic!("Can't parse safetensors: {e}"))
        })
        .collect();

    // Build lookup: tensor_name → which shard
    let mut tensors: HashMap<String, &SafeTensors<'_>> = HashMap::new();
    for st in &safetensors {
        for name in st.names() {
            tensors.insert(name.to_string(), st);
        }
    }

    println!(
        "Loading {} tensors from {} shards",
        tensors.len(),
        paths.len()
    );

    // Initialize model with random weights (will be overwritten)
    let mut model = MistralModel::new(config.clone(), device);

    // Load embedding
    let embed_weight = load_tensor::<B>(&tensors, "embed_tokens.weight", device, false);
    let embed_record = burn::nn::EmbeddingRecord {
        weight: burn::module::Param::from_tensor(embed_weight.clone()),
    };
    model.decoder.embed = model.decoder.embed.load_record(embed_record);

    // Load final norm
    let norm_weight = load_tensor_1d::<B>(&tensors, "norm.weight", device);
    model.decoder.norm = make_rms_norm(norm_weight, config.rms_norm_eps, device);

    // Load lm_head: tied to embedding for 3B, separate weight for 14B.
    if config.tie_word_embeddings {
        let lm_head_weight = embed_weight.swap_dims(0, 1); // [vocab, hidden] → [hidden, vocab]
        set_linear_weight(&mut model.decoder.lm_head, lm_head_weight);
    } else {
        // Separate lm_head weight — load and transpose like other linear layers.
        let lm_head_weight = load_tensor::<B>(&tensors, "lm_head.weight", device, true);
        set_linear_weight(&mut model.decoder.lm_head, lm_head_weight);
    }

    // Load each transformer layer
    for i in 0..config.n_layers {
        let prefix = format!("layers.{i}");
        println!("  Loading layer {i}/{}", config.n_layers);

        // Attention norm
        let attn_norm_w = load_tensor_1d::<B>(
            &tensors,
            &format!("{prefix}.input_layernorm.weight"),
            device,
        );
        model.decoder.layers[i].attn_norm = make_rms_norm(attn_norm_w, config.rms_norm_eps, device);

        // Attention projections (all need transpose: PyTorch [out, in] → Burn [in, out])
        let q_w = load_tensor::<B>(
            &tensors,
            &format!("{prefix}.self_attn.q_proj.weight"),
            device,
            true,
        );
        let k_w = load_tensor::<B>(
            &tensors,
            &format!("{prefix}.self_attn.k_proj.weight"),
            device,
            true,
        );
        let v_w = load_tensor::<B>(
            &tensors,
            &format!("{prefix}.self_attn.v_proj.weight"),
            device,
            true,
        );
        let o_w = load_tensor::<B>(
            &tensors,
            &format!("{prefix}.self_attn.o_proj.weight"),
            device,
            true,
        );

        set_linear_weight(&mut model.decoder.layers[i].attention.q_proj, q_w);
        set_linear_weight(&mut model.decoder.layers[i].attention.k_proj, k_w);
        set_linear_weight(&mut model.decoder.layers[i].attention.v_proj, v_w);
        set_linear_weight(&mut model.decoder.layers[i].attention.o_proj, o_w);

        // QK-norm (Qwen3 only)
        if config.qk_norm {
            let q_norm_w = load_tensor_1d::<B>(
                &tensors,
                &format!("{prefix}.self_attn.q_norm.weight"),
                device,
            );
            let k_norm_w = load_tensor_1d::<B>(
                &tensors,
                &format!("{prefix}.self_attn.k_norm.weight"),
                device,
            );
            model.decoder.layers[i].attention.q_norm =
                Some(make_rms_norm(q_norm_w, config.rms_norm_eps, device));
            model.decoder.layers[i].attention.k_norm =
                Some(make_rms_norm(k_norm_w, config.rms_norm_eps, device));
        }

        // FFN norm
        let ffn_norm_w = load_tensor_1d::<B>(
            &tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
            device,
        );
        model.decoder.layers[i].ffn_norm = make_rms_norm(ffn_norm_w, config.rms_norm_eps, device);

        // FFN projections (all need transpose)
        let gate_w = load_tensor::<B>(
            &tensors,
            &format!("{prefix}.mlp.gate_proj.weight"),
            device,
            true,
        );
        let up_w = load_tensor::<B>(
            &tensors,
            &format!("{prefix}.mlp.up_proj.weight"),
            device,
            true,
        );
        let down_w = load_tensor::<B>(
            &tensors,
            &format!("{prefix}.mlp.down_proj.weight"),
            device,
            true,
        );

        set_linear_weight(&mut model.decoder.layers[i].ffn.gate_proj, gate_w);
        set_linear_weight(&mut model.decoder.layers[i].ffn.up_proj, up_w);
        set_linear_weight(&mut model.decoder.layers[i].ffn.down_proj, down_w);
    }

    println!("All weights loaded.");
    model
}
