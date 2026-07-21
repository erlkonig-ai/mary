//! Load an audio file (wav/mp3/m4a/flac/ogg) into 16kHz mono f32 samples
//! suitable for `AudioFeatureExtractor::extract`. Uses symphonia for
//! decoding and rubato for sample-rate conversion.

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Load any common audio file, downmix to mono, resample to 16 kHz.
/// Returns the f32 waveform in [-1, 1] range at 16 kHz.
pub fn load_audio_16k_mono(path: &Path) -> Result<Vec<f32>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("probe: {e}"))?;
    let mut reader = probed.format;

    let track = reader.default_track().ok_or("no default track")?;
    let codec_params = track.codec_params.clone();
    let track_id = track.id;
    let src_rate = codec_params.sample_rate.ok_or("unknown sample rate")? as usize;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("decoder: {e}"))?;

    // Container-level channel count may be unknown (e.g. AAC in MP4). Pull
    // it from the first successfully-decoded packet instead.
    let mut channels: Option<usize> = codec_params.channels.map(|c| c.count());
    let mut per_ch: Vec<Vec<f32>> = Vec::new();
    loop {
        let packet = match reader.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(format!("next_packet: {e}")),
        };
        if packet.track_id() != track_id { continue; }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::IoError(_)) | Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decode: {e}")),
        };
        // Lazily determine channel count from the first decoded frame.
        if channels.is_none() {
            channels = Some(decoded.spec().channels.count());
        }
        let ch = channels.unwrap();
        if per_ch.is_empty() { per_ch = (0..ch).map(|_| Vec::new()).collect(); }

        match decoded {
            AudioBufferRef::F32(buf) => {
                for c in 0..ch { per_ch[c].extend_from_slice(buf.chan(c)); }
            }
            AudioBufferRef::S16(buf) => {
                for c in 0..ch {
                    per_ch[c].extend(buf.chan(c).iter().map(|&s| s as f32 / 32768.0));
                }
            }
            AudioBufferRef::S32(buf) => {
                for c in 0..ch {
                    per_ch[c].extend(buf.chan(c).iter().map(|&s| s as f32 / 2_147_483_648.0));
                }
            }
            other => {
                let spec = *other.spec();
                let duration = other.capacity() as u64;
                let mut fbuf = symphonia::core::audio::AudioBuffer::<f32>::new(duration, spec);
                other.convert(&mut fbuf);
                for c in 0..ch { per_ch[c].extend_from_slice(fbuf.chan(c)); }
            }
        }
    }
    let channels = channels.ok_or("no audio frames decoded")?;

    // Downmix to mono by averaging.
    let n = per_ch[0].len();
    let mut mono = vec![0.0f32; n];
    for c in 0..channels {
        let src = &per_ch[c];
        // Guard against channels of different lengths.
        let m = src.len().min(n);
        for i in 0..m { mono[i] += src[i]; }
    }
    let inv = 1.0 / channels as f32;
    for v in &mut mono { *v *= inv; }

    resample_to_16k(mono, src_rate)
}

/// Resample a mono f32 buffer from `src_rate` to 16 kHz (rubato sinc).
/// Pass-through when already 16 kHz. Shared by the file loader and the live
/// capture path (`gemma_listen` resamples each finished utterance segment).
pub fn resample_to_16k(mono: Vec<f32>, src_rate: usize) -> Result<Vec<f32>, String> {
    let dst_rate = 16_000usize;
    if src_rate == dst_rate {
        return Ok(mono);
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    // rubato SincFixedIn wants fixed input chunks; use a reasonable size.
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk = 4096usize;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, 1)
        .map_err(|e| format!("rubato init: {e}"))?;
    let delay = resampler.output_delay();
    let mut out = Vec::with_capacity((mono.len() as f64 * ratio) as usize + 1024);
    let mut i = 0;
    while i + chunk <= mono.len() {
        let waves_in = vec![mono[i..i + chunk].to_vec()];
        let waves_out = resampler.process(&waves_in, None)
            .map_err(|e| format!("rubato process: {e}"))?;
        out.extend_from_slice(&waves_out[0]);
        i += chunk;
    }
    // Final partial chunk — zero-pad to size `chunk` so SincFixedIn accepts it.
    if i < mono.len() {
        let mut tail = vec![0.0f32; chunk];
        tail[..mono.len() - i].copy_from_slice(&mono[i..]);
        let waves_in = vec![tail];
        let waves_out = resampler.process(&waves_in, None)
            .map_err(|e| format!("rubato process tail: {e}"))?;
        // Trim the output to account for the zero-padded tail.
        let valid_out = ((mono.len() - i) as f64 * ratio).ceil() as usize;
        out.extend(waves_out[0].iter().take(valid_out));
    }

    // Skip the leading resampler delay so samples align with the source.
    let skipped: Vec<f32> = out.into_iter().skip(delay).collect();
    // Truncate to expected output length.
    let expected = (mono.len() as f64 * ratio).round() as usize;
    let final_out: Vec<f32> = skipped.into_iter().take(expected).collect();
    Ok(final_out)
}
