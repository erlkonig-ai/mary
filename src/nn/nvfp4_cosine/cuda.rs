//! CUDA implementations of [`super::UpperScanner`].
//!
//! Construction uploads immutable canonical planes once. Query calls transfer
//! only prepared coordinates and one scalar result per physical row. Neither
//! implementation knows about collections, source handles, or reranking.

use std::fmt;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle as DeviceHandle;

use super::{
    FLOAT_BYTES, ScanQuery, ScanSegment, UpperScanner, add_up_nonnegative, multiply_up_nonnegative,
    next_down_f64, next_up_f64, norm_and_cast_error, read_f32, roundoff_gamma,
};

type CudaRuntime = cubecl::cuda::CudaRuntime;

const PLANE_SIZE: u32 = 32;
const THREADS: u32 = 256;

#[cfg(test)]
#[cube]
fn decode_e2m1_f64(raw: u32) -> f64 {
    let magnitude = raw & 7u32;
    let exponent = (magnitude >> 1u32) & 3u32;
    let mantissa = magnitude & 1u32;
    let mut value = if exponent == 0u32 {
        f64::cast_from(mantissa) * 0.5f64
    } else {
        let mut scale = 0.5f64;
        let mut step = 1u32;
        while step < exponent {
            scale *= 2.0f64;
            step += 1u32;
        }
        f64::cast_from(2u32 + mantissa) * scale
    };
    if raw & 8u32 != 0u32 {
        value = -value;
    }
    value
}

#[cfg(test)]
#[cube]
fn decode_e4m3_f64(raw: u32) -> f64 {
    let exponent = (raw >> 3u32) & 15u32;
    let mantissa = raw & 7u32;
    if exponent == 0u32 {
        f64::cast_from(mantissa) * 0.001953125f64
    } else {
        let mut scale = 0.0009765625f64;
        let mut step = 0u32;
        while step < exponent {
            scale *= 2.0f64;
            step += 1u32;
        }
        f64::cast_from(8u32 + mantissa) * scale
    }
}

#[cube]
fn decode_e2m1_f32(raw: u32) -> f32 {
    let magnitude = raw & 7u32;
    let exponent = (magnitude >> 1u32) & 3u32;
    let mantissa = magnitude & 1u32;
    let mut value = if exponent == 0u32 {
        f32::cast_from(mantissa) * 0.5f32
    } else {
        let mut scale = 0.5f32;
        let mut step = 1u32;
        while step < exponent {
            scale *= 2.0f32;
            step += 1u32;
        }
        f32::cast_from(2u32 + mantissa) * scale
    };
    if raw & 8u32 != 0u32 {
        value = -value;
    }
    value
}

#[cube]
fn decode_e4m3_f32(raw: u32) -> f32 {
    let exponent = (raw >> 3u32) & 15u32;
    let mantissa = raw & 7u32;
    if exponent == 0u32 {
        f32::cast_from(mantissa) * 0.001953125f32
    } else {
        let mut scale = 0.0009765625f32;
        let mut step = 0u32;
        while step < exponent {
            scale *= 2.0f32;
            step += 1u32;
        }
        f32::cast_from(8u32 + mantissa) * scale
    }
}

#[cfg(test)]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn decode_dot_f64(
    query: &Array<f64>,
    primary_globals: &Array<f32>,
    primary_scales: &Array<u8>,
    primary_codes: &Array<u8>,
    correction_globals: &Array<f32>,
    correction_scales: &Array<u8>,
    correction_codes: &Array<u8>,
    dots: &mut Array<f64>,
    rows: u32,
    blocks_per_row: u32,
    codes_per_row: u32,
    output_row_offset: u32,
) {
    let row = (ABSOLUTE_POS as u32) / PLANE_SIZE;
    if row < rows {
        let lane = UNIT_POS_PLANE;
        let primary_global = f64::cast_from(primary_globals[row as usize]);
        let correction_global = f64::cast_from(correction_globals[row as usize]);
        let mut partial = 0.0f64;
        let mut code = lane;
        while code < codes_per_row {
            let block = code / 8u32;
            let scale_index = row * blocks_per_row + block;
            let primary_scale = primary_global
                * decode_e4m3_f64(u32::cast_from(primary_scales[scale_index as usize]));
            let correction_scale = correction_global
                * decode_e4m3_f64(u32::cast_from(correction_scales[scale_index as usize]));
            let code_index = row * codes_per_row + code;
            let primary = u32::cast_from(primary_codes[code_index as usize]);
            let correction = u32::cast_from(correction_codes[code_index as usize]);
            let low = decode_e2m1_f64(primary & 15u32) * primary_scale
                + decode_e2m1_f64(correction & 15u32) * correction_scale;
            let high = decode_e2m1_f64(primary >> 4u32) * primary_scale
                + decode_e2m1_f64(correction >> 4u32) * correction_scale;
            let coordinate = code * 2u32;
            partial += query[coordinate as usize] * low;
            partial += query[(coordinate + 1u32) as usize] * high;
            code += PLANE_SIZE;
        }
        let dot = plane_sum(partial);
        if lane == 0u32 {
            dots[(output_row_offset + row) as usize] = dot;
        }
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
/// Prescribed binary32 decode and fixed reduction tree.
///
/// The proof requires RN-even `fma.f32`, gradual underflow (no `.ftz`), and
/// exactly five XOR-shuffle additions. Fast math invalidates this kernel.
fn decode_dot_f32(
    query: &Array<f32>,
    primary_globals: &Array<f32>,
    primary_scales: &Array<u8>,
    primary_codes: &Array<u8>,
    correction_globals: &Array<f32>,
    correction_scales: &Array<u8>,
    correction_codes: &Array<u8>,
    dots: &mut Array<f32>,
    rows: u32,
    blocks_per_row: u32,
    codes_per_row: u32,
    output_row_offset: u32,
) {
    let row = (ABSOLUTE_POS as u32) / PLANE_SIZE;
    if row < rows {
        let lane = UNIT_POS_PLANE;
        let primary_global = primary_globals[row as usize];
        let correction_global = correction_globals[row as usize];
        let mut partial = 0.0f32;
        let mut code = lane;
        while code < codes_per_row {
            let block = code / 8u32;
            let scale_index = row * blocks_per_row + block;
            let primary_scale = fma(
                primary_global,
                decode_e4m3_f32(u32::cast_from(primary_scales[scale_index as usize])),
                0.0f32,
            );
            let correction_scale = fma(
                correction_global,
                decode_e4m3_f32(u32::cast_from(correction_scales[scale_index as usize])),
                0.0f32,
            );
            let code_index = row * codes_per_row + code;
            let primary = u32::cast_from(primary_codes[code_index as usize]);
            let correction = u32::cast_from(correction_codes[code_index as usize]);
            let primary_low = fma(decode_e2m1_f32(primary & 15u32), primary_scale, 0.0f32);
            let primary_high = fma(decode_e2m1_f32(primary >> 4u32), primary_scale, 0.0f32);
            let low = fma(
                decode_e2m1_f32(correction & 15u32),
                correction_scale,
                primary_low,
            );
            let high = fma(
                decode_e2m1_f32(correction >> 4u32),
                correction_scale,
                primary_high,
            );
            let coordinate = code * 2u32;
            partial = fma(query[coordinate as usize], low, partial);
            partial = fma(query[(coordinate + 1u32) as usize], high, partial);
            code += PLANE_SIZE;
        }
        partial += plane_shuffle_xor(partial, 1u32);
        partial += plane_shuffle_xor(partial, 2u32);
        partial += plane_shuffle_xor(partial, 4u32);
        partial += plane_shuffle_xor(partial, 8u32);
        partial += plane_shuffle_xor(partial, 16u32);
        if lane == 0u32 {
            dots[(output_row_offset + row) as usize] = partial;
        }
    }
}

#[derive(Clone)]
struct ResidentStage {
    globals: DeviceHandle,
    block_scales: DeviceHandle,
    codes: DeviceHandle,
}

#[derive(Clone)]
struct ResidentSegment {
    identity: [u8; 32],
    rows: u32,
    blocks_per_row: u32,
    codes_per_row: u32,
    stages: Option<[ResidentStage; 2]>,
}

/// Failure to construct or execute a resident CUDA NVFP4 scan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaError {
    GeometryOverflow(&'static str),
    UnsupportedPlane {
        min: u32,
        max: u32,
    },
    SegmentMismatch {
        index: usize,
    },
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    Device(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GeometryOverflow(what) => write!(formatter, "NVFP4 CUDA {what} exceeds u32"),
            Self::UnsupportedPlane { min, max } => write!(
                formatter,
                "NVFP4 CUDA scan needs 32-lane planes, device reports {min}..={max}"
            ),
            Self::SegmentMismatch { index } => {
                write!(
                    formatter,
                    "NVFP4 CUDA resident segment {index} does not match"
                )
            }
            Self::ShapeMismatch {
                what,
                expected,
                actual,
            } => write!(
                formatter,
                "NVFP4 CUDA {what} has length {actual}, expected {expected}"
            ),
            Self::Device(error) => write!(formatter, "NVFP4 CUDA operation failed: {error}"),
        }
    }
}

impl std::error::Error for CudaError {}

struct Resident {
    client: ComputeClient<CudaRuntime>,
    segments: Vec<ResidentSegment>,
    physical_dimension: usize,
    physical_rows: usize,
}

impl Resident {
    fn new(
        source: &[ScanSegment<'_>],
        device: &cubecl::cuda::CudaDevice,
    ) -> Result<Self, CudaError> {
        use cubecl::ir::features::Plane;

        let client = CudaRuntime::client(device);
        let properties = client.properties();
        let (min, max) = (
            properties.hardware.plane_size_min,
            properties.hardware.plane_size_max,
        );
        if !properties.features.plane.contains(Plane::Ops) || min != PLANE_SIZE || max != PLANE_SIZE
        {
            return Err(CudaError::UnsupportedPlane { min, max });
        }

        let physical_dimension = source
            .first()
            .map(|segment| segment.codes_per_row() * 2)
            .unwrap_or(0);
        let mut physical_rows = 0u32;
        let mut segments = Vec::with_capacity(source.len());
        for &segment in source {
            let rows = u32::try_from(segment.rows())
                .map_err(|_| CudaError::GeometryOverflow("row count"))?;
            let blocks_per_row = u32::try_from(segment.blocks_per_row())
                .map_err(|_| CudaError::GeometryOverflow("block count"))?;
            let codes_per_row = u32::try_from(segment.codes_per_row())
                .map_err(|_| CudaError::GeometryOverflow("code count"))?;
            rows.checked_mul(PLANE_SIZE)
                .and_then(|threads| threads.checked_add(THREADS - 1))
                .ok_or(CudaError::GeometryOverflow("thread count"))?;
            rows.checked_mul(blocks_per_row)
                .ok_or(CudaError::GeometryOverflow("scale-plane index"))?;
            rows.checked_mul(codes_per_row)
                .ok_or(CudaError::GeometryOverflow("code-plane index"))?;
            physical_rows = physical_rows
                .checked_add(rows)
                .ok_or(CudaError::GeometryOverflow("physical row count"))?;
            let stages = if rows == 0 {
                None
            } else {
                Some(segment.stages().map(|stage| {
                    let globals: Vec<_> = stage
                        .global_scale_bytes()
                        .chunks_exact(FLOAT_BYTES)
                        .map(read_f32)
                        .collect();
                    ResidentStage {
                        globals: client.create_from_slice(f32::as_bytes(&globals)),
                        block_scales: client.create_from_slice(stage.block_scales()),
                        codes: client.create_from_slice(stage.codes()),
                    }
                }))
            };
            segments.push(ResidentSegment {
                identity: segment.identity(),
                rows,
                blocks_per_row,
                codes_per_row,
                stages,
            });
        }
        Ok(Self {
            client,
            segments,
            physical_dimension,
            physical_rows: physical_rows as usize,
        })
    }

    fn validate(
        &self,
        query_len: usize,
        segments: &[ScanSegment<'_>],
        output_len: usize,
    ) -> Result<(), CudaError> {
        if segments.len() != self.segments.len() {
            return Err(CudaError::ShapeMismatch {
                what: "segment sequence",
                expected: self.segments.len(),
                actual: segments.len(),
            });
        }
        for (index, (resident, source)) in self.segments.iter().zip(segments).enumerate() {
            if resident.identity != source.identity() {
                return Err(CudaError::SegmentMismatch { index });
            }
        }
        if output_len != self.physical_rows {
            return Err(CudaError::ShapeMismatch {
                what: "output",
                expected: self.physical_rows,
                actual: output_len,
            });
        }
        if query_len != self.physical_dimension {
            return Err(CudaError::ShapeMismatch {
                what: "query",
                expected: self.physical_dimension,
                actual: query_len,
            });
        }
        Ok(())
    }
}

fn empty_scan(resident: &Resident, upper_raw_dots: &[f64]) -> Result<bool, CudaError> {
    if resident.physical_rows != 0 {
        return Ok(false);
    }
    if !upper_raw_dots.is_empty() {
        return Err(CudaError::ShapeMismatch {
            what: "output",
            expected: 0,
            actual: upper_raw_dots.len(),
        });
    }
    Ok(true)
}

#[cfg(test)]
fn launch_f64(resident: &Resident, coordinates: &[f64]) -> Result<Vec<u8>, CudaError> {
    let query = resident
        .client
        .create_from_slice(f64::as_bytes(coordinates));
    let cube_dim = CubeDim::new_1d(THREADS);
    let output_bytes = resident
        .physical_rows
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(CudaError::GeometryOverflow("output byte count"))?;
    let output = resident.client.empty(output_bytes);
    let mut offset = 0u32;
    for segment in &resident.segments {
        let Some(stages) = &segment.stages else {
            continue;
        };
        let threads = segment.rows as usize * PLANE_SIZE as usize;
        let dispatch = cubecl::calculate_cube_count_elemwise(&resident.client, threads, cube_dim);
        unsafe {
            decode_dot_f64::launch_unchecked::<CudaRuntime>(
                &resident.client,
                dispatch,
                cube_dim,
                ArrayArg::from_raw_parts(query.clone(), coordinates.len()),
                ArrayArg::from_raw_parts(stages[0].globals.clone(), segment.rows as usize),
                ArrayArg::from_raw_parts(
                    stages[0].block_scales.clone(),
                    segment.rows as usize * segment.blocks_per_row as usize,
                ),
                ArrayArg::from_raw_parts(
                    stages[0].codes.clone(),
                    segment.rows as usize * segment.codes_per_row as usize,
                ),
                ArrayArg::from_raw_parts(stages[1].globals.clone(), segment.rows as usize),
                ArrayArg::from_raw_parts(
                    stages[1].block_scales.clone(),
                    segment.rows as usize * segment.blocks_per_row as usize,
                ),
                ArrayArg::from_raw_parts(
                    stages[1].codes.clone(),
                    segment.rows as usize * segment.codes_per_row as usize,
                ),
                ArrayArg::from_raw_parts(output.clone(), resident.physical_rows),
                segment.rows,
                segment.blocks_per_row,
                segment.codes_per_row,
                offset,
            )
        };
        offset += segment.rows;
    }
    resident
        .client
        .read_one(output)
        .map(|bytes| bytes.to_vec())
        .map_err(|error| CudaError::Device(format!("{error:?}")))
}

fn launch_f32(resident: &Resident, coordinates: &[f32]) -> Result<Vec<u8>, CudaError> {
    let query = resident
        .client
        .create_from_slice(f32::as_bytes(coordinates));
    let cube_dim = CubeDim::new_1d(THREADS);
    let output_bytes = resident
        .physical_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CudaError::GeometryOverflow("output byte count"))?;
    let output = resident.client.empty(output_bytes);
    let mut offset = 0u32;
    for segment in &resident.segments {
        let Some(stages) = &segment.stages else {
            continue;
        };
        let threads = segment.rows as usize * PLANE_SIZE as usize;
        let dispatch = cubecl::calculate_cube_count_elemwise(&resident.client, threads, cube_dim);
        unsafe {
            decode_dot_f32::launch_unchecked::<CudaRuntime>(
                &resident.client,
                dispatch,
                cube_dim,
                ArrayArg::from_raw_parts(query.clone(), coordinates.len()),
                ArrayArg::from_raw_parts(stages[0].globals.clone(), segment.rows as usize),
                ArrayArg::from_raw_parts(
                    stages[0].block_scales.clone(),
                    segment.rows as usize * segment.blocks_per_row as usize,
                ),
                ArrayArg::from_raw_parts(
                    stages[0].codes.clone(),
                    segment.rows as usize * segment.codes_per_row as usize,
                ),
                ArrayArg::from_raw_parts(stages[1].globals.clone(), segment.rows as usize),
                ArrayArg::from_raw_parts(
                    stages[1].block_scales.clone(),
                    segment.rows as usize * segment.blocks_per_row as usize,
                ),
                ArrayArg::from_raw_parts(
                    stages[1].codes.clone(),
                    segment.rows as usize * segment.codes_per_row as usize,
                ),
                ArrayArg::from_raw_parts(output.clone(), resident.physical_rows),
                segment.rows,
                segment.blocks_per_row,
                segment.codes_per_row,
                offset,
            )
        };
        offset += segment.rows;
    }
    resident
        .client
        .read_one(output)
        .map(|bytes| bytes.to_vec())
        .map_err(|error| CudaError::Device(format!("{error:?}")))
}

/// Test-only f64 CUDA proof oracle for immutable NVFP4 planes.
#[cfg(test)]
struct CudaF64UpperScanner {
    resident: Resident,
}

#[cfg(test)]
impl CudaF64UpperScanner {
    fn new(
        segments: &[ScanSegment<'_>],
        device: &cubecl::cuda::CudaDevice,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            resident: Resident::new(segments, device)?,
        })
    }
}

#[cfg(test)]
impl UpperScanner for CudaF64UpperScanner {
    type Error = CudaError;

    fn scan_upper(
        &self,
        query: ScanQuery<'_>,
        segments: &[ScanSegment<'_>],
        upper_raw_dots: &mut [f64],
    ) -> Result<(), Self::Error> {
        if empty_scan(&self.resident, upper_raw_dots)? {
            return Ok(());
        }
        let coordinates = query.coordinates();
        self.resident
            .validate(coordinates.len(), segments, upper_raw_dots.len())?;
        let bytes = launch_f64(&self.resident, coordinates)?;
        let expected = self.resident.physical_rows * std::mem::size_of::<f64>();
        if bytes.len() != expected {
            return Err(CudaError::ShapeMismatch {
                what: "readback bytes",
                expected,
                actual: bytes.len(),
            });
        }
        let gamma = roundoff_gamma(coordinates.len().saturating_mul(2), f64::EPSILON / 2.0);
        let underflow = multiply_up_nonnegative(
            coordinates.len().saturating_mul(2) as f64,
            f64::from_bits(1),
        );
        let mut raw = bytes.chunks_exact(std::mem::size_of::<f64>());
        let mut output = 0;
        for &segment in segments {
            for row in 0..segment.rows() {
                let center = f64::from_ne_bytes(
                    raw.next()
                        .expect("one result per row")
                        .try_into()
                        .expect("eight-byte f64"),
                );
                if !center.is_finite() {
                    upper_raw_dots[output] = f64::INFINITY;
                } else {
                    let norm = segment
                        .row_certificate(row)
                        .expect("validated segment certificate")
                        .reconstruction_norm();
                    let roundoff = multiply_up_nonnegative(
                        multiply_up_nonnegative(gamma, query.norm_bound()),
                        norm,
                    );
                    upper_raw_dots[output] =
                        next_up_f64(center + add_up_nonnegative(roundoff, underflow));
                }
                output += 1;
            }
        }
        Ok(())
    }
}

/// Certified ordinary-f32 CUDA completion for immutable NVFP4 planes.
pub struct CudaUpperScanner {
    resident: Resident,
    accumulation_gamma: f64,
    underflow_allowance: f64,
}

impl CudaUpperScanner {
    /// Upload the immutable planes to the Spark's CUDA device.
    pub fn new(segments: &[ScanSegment<'_>]) -> Result<Self, CudaError> {
        let device = cubecl::cuda::CudaDevice::default();
        let resident = Resident::new(segments, &device)?;
        let lane_fmas = resident.physical_dimension.div_ceil(PLANE_SIZE as usize);
        let operations = lane_fmas.saturating_add(5);
        let unit = f64::from(f32::EPSILON) / 2.0;
        let accumulation_gamma = roundoff_gamma(operations, unit);
        let contributing_nodes = resident
            .physical_dimension
            .saturating_add(PLANE_SIZE as usize - 1);
        let half_min_subnormal = f64::from(f32::from_bits(1)) / 2.0;
        let propagation = next_down_f64(1.0 - multiply_up_nonnegative(operations as f64, unit));
        let underflow_allowance = if propagation <= 0.0 {
            f64::INFINITY
        } else {
            next_up_f64(
                multiply_up_nonnegative(contributing_nodes as f64, half_min_subnormal)
                    / propagation,
            )
        };
        Ok(Self {
            resident,
            accumulation_gamma,
            underflow_allowance,
        })
    }
}

impl UpperScanner for CudaUpperScanner {
    type Error = CudaError;

    fn scan_upper(
        &self,
        query: ScanQuery<'_>,
        segments: &[ScanSegment<'_>],
        upper_raw_dots: &mut [f64],
    ) -> Result<(), Self::Error> {
        if empty_scan(&self.resident, upper_raw_dots)? {
            return Ok(());
        }
        let query64 = query.coordinates();
        self.resident
            .validate(query64.len(), segments, upper_raw_dots.len())?;
        let query32: Vec<_> = query64.iter().map(|&value| value as f32).collect();
        let (query32_norm, query_cast_error) =
            norm_and_cast_error(query64, &query32).expect("matching query dimensions");
        let bytes = launch_f32(&self.resident, &query32)?;
        let expected = self.resident.physical_rows * std::mem::size_of::<f32>();
        if bytes.len() != expected {
            return Err(CudaError::ShapeMismatch {
                what: "readback bytes",
                expected,
                actual: bytes.len(),
            });
        }
        let mut raw = bytes.chunks_exact(std::mem::size_of::<f32>());
        let mut output = 0;
        for &segment in segments {
            for row in 0..segment.rows() {
                let center = f64::from(f32::from_ne_bytes(
                    raw.next()
                        .expect("one result per row")
                        .try_into()
                        .expect("four-byte f32"),
                ));
                if !center.is_finite() {
                    upper_raw_dots[output] = f64::INFINITY;
                } else {
                    let certificate = segment
                        .row_certificate(row)
                        .expect("validated segment certificate");
                    let norm = certificate.reconstruction_norm();
                    let decode_error = certificate.error_bound();
                    let cast = multiply_up_nonnegative(query_cast_error, norm);
                    let decode = multiply_up_nonnegative(query32_norm, decode_error);
                    let decoded_norm = add_up_nonnegative(norm, decode_error);
                    let accumulation = multiply_up_nonnegative(
                        multiply_up_nonnegative(self.accumulation_gamma, query32_norm),
                        decoded_norm,
                    );
                    let error = add_up_nonnegative(
                        add_up_nonnegative(cast, decode),
                        add_up_nonnegative(accumulation, self.underflow_allowance),
                    );
                    let upper = next_up_f64(center + error);
                    upper_raw_dots[output] = if upper.is_nan() { f64::INFINITY } else { upper };
                }
                output += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        CandidateCertificate, PreparedQuery, QUANT_BLOCK, QuantizedRow, QuantizedStage,
        ROTATION_BLOCK, ScanStage, decode_f32_reconstruction, decode_stage, exact_cosine,
        outward_l2, outward_norm, raw_dot_f64, upward_f32,
    };
    use super::*;

    use std::collections::BTreeSet;
    use std::fs;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    use cubecl::config::RuntimeConfig;
    use cubecl::config::cache::CacheConfig;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct CachedPtxEntry {
        #[allow(dead_code)]
        key: ciborium::Value,
        value: CachedPtx,
    }

    #[derive(Deserialize)]
    struct CachedPtx {
        entrypoint_name: String,
        #[allow(dead_code)]
        shared_mem_bytes: usize,
        ptx: Vec<i8>,
    }

    fn ptx_contract(ptx: &str) -> Result<(), String> {
        if !ptx.contains(".visible .entry decode_dot_f32(") {
            return Err("missing decode_dot_f32 entry point".into());
        }
        if !ptx.contains(".target sm_121a") {
            return Err("gate did not compile for the Spark sm_121a target".into());
        }
        for forbidden in [".ftz", ".approx", "--use_fast_math"] {
            if ptx.contains(forbidden) {
                return Err(format!("forbidden PTX arithmetic modifier {forbidden}"));
            }
        }

        let mut fmas = 0usize;
        let mut adds = 0usize;
        let mut shuffles = Vec::new();
        let mut shuffle_waiting_for_add = false;
        let mut shuffle_adds = 0usize;
        for line in ptx.lines() {
            let mut words = line.trim().split_ascii_whitespace();
            let mut opcode = words.next().unwrap_or_default();
            if opcode.starts_with('@') {
                opcode = words.next().unwrap_or_default();
            }
            opcode = opcode.trim_end_matches(';');
            if opcode.starts_with("fma.") && opcode.ends_with(".f32") {
                if opcode != "fma.rn.f32" {
                    return Err(format!("non-RN f32 FMA in PTX: {opcode}"));
                }
                fmas += 1;
            }
            if opcode.starts_with("mad.") && opcode.ends_with(".f32") {
                return Err(format!("contract-breaking f32 MAD in PTX: {opcode}"));
            }
            if opcode.starts_with("add.") && opcode.ends_with(".f32") {
                // PTX spells the default RN-even form both as `add.f32` and
                // `add.rn.f32`; every other explicit rounding mode is wrong.
                if opcode != "add.f32" && opcode != "add.rn.f32" {
                    return Err(format!("non-RN f32 add in PTX: {opcode}"));
                }
                adds += 1;
                if shuffle_waiting_for_add {
                    shuffle_waiting_for_add = false;
                    shuffle_adds += 1;
                }
            }
            if opcode.starts_with("mul.")
                && opcode.ends_with(".f32")
                && opcode != "mul.f32"
                && opcode != "mul.rn.f32"
            {
                return Err(format!("non-RN f32 multiply in PTX: {opcode}"));
            }
            if opcode == "shfl.sync.bfly.b32" {
                if shuffle_waiting_for_add {
                    return Err("butterfly shuffle was not completed by an f32 add".into());
                }
                let fields: Vec<_> = line.split(',').map(str::trim).collect();
                let mask = fields
                    .get(2)
                    .and_then(|field| field.parse::<u32>().ok())
                    .ok_or_else(|| format!("cannot read butterfly mask from `{line}`"))?;
                shuffles.push(mask);
                shuffle_waiting_for_add = true;
            }
        }
        if fmas < 8 {
            return Err(format!(
                "only {fmas} RN-even f32 FMAs remain in the decode/dot path"
            ));
        }
        if adds < 5 {
            return Err(format!(
                "only {adds} RN-even f32 additions remain in the reduction"
            ));
        }
        if shuffles != [1, 2, 4, 8, 16] {
            return Err(format!(
                "reduction is not the prescribed five-level XOR tree: {shuffles:?}"
            ));
        }
        if shuffle_waiting_for_add || shuffle_adds != 5 {
            return Err(format!(
                "only {shuffle_adds} of five butterfly shuffles feed an f32 add"
            ));
        }
        Ok(())
    }

    fn collect_ptx_files(root: &Path, files: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(root).unwrap_or_else(|error| {
            panic!(
                "cannot read CubeCL cache directory {}: {error}",
                root.display()
            )
        }) {
            let path = entry.expect("CubeCL cache directory entry").path();
            if path.is_dir() {
                collect_ptx_files(&path, files);
            } else if path.file_name().is_some_and(|name| name == "chunk0.cbor") {
                files.push(path);
            }
        }
    }

    fn emitted_f32_ptx(cache_root: &Path) -> String {
        let mut files = Vec::new();
        collect_ptx_files(cache_root, &mut files);
        for path in files {
            let bytes = fs::read(&path).expect("read CubeCL PTX cache chunk");
            let mut cursor = Cursor::new(bytes);
            while (cursor.position() as usize) < cursor.get_ref().len() {
                let before = cursor.position();
                let entry: CachedPtxEntry =
                    ciborium::from_reader(&mut cursor).unwrap_or_else(|error| {
                        panic!("decode {} at {before}: {error}", path.display())
                    });
                if entry.value.entrypoint_name == "decode_dot_f32" {
                    let bytes: Vec<_> = entry
                        .value
                        .ptx
                        .into_iter()
                        .map(|byte| byte as u8)
                        .take_while(|&byte| byte != 0)
                        .collect();
                    return String::from_utf8(bytes).expect("NVRTC emits ASCII PTX");
                }
            }
        }
        panic!(
            "CubeCL cache under {} contains no decode_dot_f32 PTX",
            cache_root.display()
        );
    }

    fn certified_row(stages: [QuantizedStage; 2]) -> QuantizedRow {
        let primary = decode_stage(&stages[0]);
        let correction = decode_stage(&stages[1]);
        let canonical: Vec<_> = primary
            .into_iter()
            .zip(correction)
            .map(|(primary, correction)| primary + correction)
            .collect();
        let explicit_f32 = decode_f32_reconstruction(&stages[0], &stages[1]);
        let norm = upward_f32(outward_norm(&canonical).unwrap()).unwrap();
        let error = upward_f32(outward_l2(&canonical, &explicit_f32).unwrap()).unwrap();
        QuantizedRow {
            stages,
            norm: norm.to_le_bytes(),
            error: error.to_le_bytes(),
        }
    }

    fn packed_codes(offset: usize) -> Vec<u8> {
        (0..(ROTATION_BLOCK / 2))
            .map(|index| {
                let low = ((offset + 2 * index) & 15) as u8;
                let high = ((offset + 2 * index + 1) & 15) as u8;
                low | (high << 4)
            })
            .collect()
    }

    fn adversarial_rows() -> Vec<QuantizedRow> {
        let globals = [
            (1.0f32, f32::from_bits(1.0f32.to_bits() - 1)),
            (f32::MIN_POSITIVE, f32::MIN_POSITIVE),
            (f32::from_bits(1), f32::from_bits(1)),
            (2.0f32.powi(-40), 2.0f32.powi(-40)),
        ];
        let mut rows = Vec::new();
        for (row, (primary_global, correction_global)) in globals.into_iter().enumerate() {
            let stages = std::array::from_fn(|stage| {
                let block_scales = (0..(ROTATION_BLOCK / QUANT_BLOCK))
                    .map(|block| (row * 32 + stage * 16 + block).min(0x7e) as u8)
                    .collect();
                QuantizedStage {
                    global: if stage == 0 {
                        primary_global.to_le_bytes()
                    } else {
                        correction_global.to_le_bytes()
                    },
                    block_scales,
                    codes: packed_codes(row * 3 + stage),
                }
            });
            rows.push(certified_row(stages));
        }

        let cancelling = QuantizedStage {
            global: 1.0f32.to_le_bytes(),
            block_scales: vec![0x7e; ROTATION_BLOCK / QUANT_BLOCK],
            codes: vec![0x77; ROTATION_BLOCK / 2],
        };
        let correction = QuantizedStage {
            global: f32::from_bits(1.0f32.to_bits() - 1).to_le_bytes(),
            block_scales: vec![0x7e; ROTATION_BLOCK / QUANT_BLOCK],
            codes: vec![0xff; ROTATION_BLOCK / 2],
        };
        rows.push(certified_row([cancelling, correction]));

        let maximal_zero = QuantizedStage {
            global: f32::MAX.to_le_bytes(),
            block_scales: vec![0; ROTATION_BLOCK / QUANT_BLOCK],
            codes: vec![0x77; ROTATION_BLOCK / 2],
        };
        let zero = QuantizedStage {
            global: 0.0f32.to_le_bytes(),
            block_scales: vec![0; ROTATION_BLOCK / QUANT_BLOCK],
            codes: vec![0; ROTATION_BLOCK / 2],
        };
        rows.push(certified_row([maximal_zero, zero]));
        rows
    }

    fn adversarial_queries() -> Vec<Vec<f64>> {
        let one = f64::from(1.0f32);
        let one_next = f64::from(f32::from_bits(1.0f32.to_bits() + 1));
        let next_next = f64::from(f32::from_bits(1.0f32.to_bits() + 2));
        let half_subnormal = f64::from(f32::from_bits(1)) / 2.0;
        vec![
            vec![0.0; ROTATION_BLOCK],
            (0..ROTATION_BLOCK)
                .map(|index| match index % 4 {
                    0 => (one + one_next) / 2.0,
                    1 => -((one_next + next_next) / 2.0),
                    2 => 1.0,
                    _ => -1.0,
                })
                .collect(),
            (0..ROTATION_BLOCK)
                .map(|index| match index % 4 {
                    0 => half_subnormal,
                    1 => -half_subnormal,
                    2 => f64::from(f32::from_bits(1)),
                    _ => -f64::from(f32::from_bits(1)),
                })
                .collect(),
            (0..ROTATION_BLOCK)
                .map(|index| if index == 0 { f64::from(f32::MAX) } else { 0.0 })
                .collect(),
        ]
    }

    fn vector(row: usize, dimension: usize) -> Vec<f32> {
        (0..dimension)
            .map(|coordinate| {
                let phase = row as f32 * 0.37 + coordinate as f32 * 0.19;
                phase.sin() + (phase * 0.31).cos()
            })
            .collect()
    }

    fn stage_planes<'a>(
        rows: &'a [QuantizedRow],
        globals: &'a mut [Vec<u8>; 2],
        scales: &'a mut [Vec<u8>; 2],
        codes: &'a mut [Vec<u8>; 2],
    ) -> [ScanStage<'a>; 2] {
        for row in rows {
            for stage_index in 0..2 {
                let stage = &row.stages()[stage_index];
                globals[stage_index].extend_from_slice(stage.global_scale_bytes());
                scales[stage_index].extend_from_slice(stage.block_scales());
                codes[stage_index].extend_from_slice(stage.codes());
            }
        }
        std::array::from_fn(|stage| ScanStage::new(&globals[stage], &scales[stage], &codes[stage]))
    }

    #[test]
    fn ptx_gate_rejects_contract_drift() {
        let mut ptx =
            String::from(".version 9.0\n.target sm_121a\n.visible .entry decode_dot_f32(\n");
        for _ in 0..8 {
            ptx.push_str("fma.rn.f32 %f1, %f2, %f3, %f4;\n");
        }
        for mask in [1, 2, 4, 8, 16] {
            ptx.push_str(&format!(
                "shfl.sync.bfly.b32 %r1|%p1, %r2, {mask}, 31, -1;\nadd.f32 %f1, %f2, %f3;\n"
            ));
        }
        assert_eq!(ptx_contract(&ptx), Ok(()));
        assert!(ptx_contract(&ptx.replace("add.f32", "add.ftz.f32")).is_err());
        assert!(ptx_contract(&ptx.replace("fma.rn.f32", "fma.rz.f32")).is_err());
        assert!(
            ptx_contract(&ptx.replacen(
                "shfl.sync.bfly.b32 %r1|%p1, %r2, 1, 31, -1;\nadd.f32",
                "shfl.sync.bfly.b32 %r1|%p1, %r2, 1, 31, -1;\nmov.f32",
                1,
            ))
            .is_err()
        );
        assert!(
            ptx_contract(&ptx.replace("shfl.sync.bfly.b32 %r1|%p1, %r2, 16, 31, -1;\n", ""))
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires the Spark CUDA device and inspects its emitted PTX"]
    fn cuda_f32_adversarial_certificate_and_ptx_gate() {
        const CHILD: &str = "MARY_NVFP4_PTX_GATE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "nn::nvfp4_cosine::cuda::tests::cuda_f32_adversarial_certificate_and_ptx_gate",
                    "--ignored",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .status()
                .expect("launch isolated PTX-gate test process");
            assert!(status.success(), "isolated PTX-gate process failed");
            return;
        }

        let cache_root =
            std::env::temp_dir().join(format!("mary-nvfp4-ptx-contract-{}", std::process::id()));
        let _ = fs::remove_dir_all(&cache_root);
        let mut config = cubecl::config::CubeClRuntimeConfig::default();
        config.compilation.cache = Some(CacheConfig::File(cache_root.clone()));
        cubecl::config::CubeClRuntimeConfig::set(config);

        let rows = adversarial_rows();
        let e4m3: BTreeSet<_> = rows[..4]
            .iter()
            .flat_map(|row| row.stages())
            .flat_map(QuantizedStage::block_scales)
            .copied()
            .collect();
        assert_eq!(e4m3, (0u8..=0x7e).collect());
        let e2m1: BTreeSet<_> = rows
            .iter()
            .flat_map(|row| row.stages())
            .flat_map(QuantizedStage::codes)
            .flat_map(|&packed| [packed & 15, packed >> 4])
            .collect();
        assert_eq!(e2m1, (0u8..=15).collect());
        assert!(rows.iter().any(|row| {
            row.stages()
                .iter()
                .any(|stage| read_f32(stage.global_scale_bytes()) == f32::from_bits(1))
        }));
        assert!(rows.iter().any(|row| {
            row.stages()
                .iter()
                .any(|stage| read_f32(stage.global_scale_bytes()) == f32::MAX)
        }));

        let mut globals = [Vec::new(), Vec::new()];
        let mut scales = [Vec::new(), Vec::new()];
        let mut codes = [Vec::new(), Vec::new()];
        let stages = stage_planes(&rows, &mut globals, &mut scales, &mut codes);
        let mut norms = Vec::with_capacity(rows.len() * FLOAT_BYTES);
        let mut errors = Vec::with_capacity(rows.len() * FLOAT_BYTES);
        for row in &rows {
            norms.extend_from_slice(row.reconstruction_norm_bytes());
            errors.extend_from_slice(row.error_bound_bytes());
        }
        let segment = ScanSegment::new(
            [0xa5; 32],
            rows.len(),
            ROTATION_BLOCK,
            ROTATION_BLOCK / QUANT_BLOCK,
            ROTATION_BLOCK / 2,
            stages,
            &norms,
            &errors,
        )
        .unwrap();
        let scanner = CudaUpperScanner::new(&[segment]).unwrap();
        for (query_index, coordinates) in adversarial_queries().iter().enumerate() {
            let query = ScanQuery {
                coordinates,
                norm_bound: outward_norm(coordinates).unwrap(),
            };
            let mut uppers = vec![f64::NAN; rows.len()];
            scanner.scan_upper(query, &[segment], &mut uppers).unwrap();
            let mut finite = 0;
            for (row, &upper) in uppers.iter().enumerate() {
                assert!(!upper.is_nan(), "query {query_index}, row {row}: NaN upper");
                if upper.is_finite() {
                    finite += 1;
                } else {
                    assert_eq!(upper, f64::INFINITY);
                }
                let canonical = raw_dot_f64(coordinates, segment, row);
                if canonical.is_finite() {
                    assert!(
                        canonical <= upper,
                        "query {query_index}, row {row}: canonical {canonical:e} > upper {upper:e}"
                    );
                } else {
                    assert_eq!(upper, f64::INFINITY);
                }
            }
            assert!(finite > 0, "query {query_index} failed open for every row");
        }

        // A syntactically valid but numerically extreme plane must never turn
        // device overflow into a finite pruning certificate.
        let maximal = f32::MAX.to_le_bytes();
        let zero = 0.0f32.to_le_bytes();
        let maximal_scales = vec![0x7e; ROTATION_BLOCK / QUANT_BLOCK];
        let zero_scales = vec![0; ROTATION_BLOCK / QUANT_BLOCK];
        let maximal_codes = vec![0x77; ROTATION_BLOCK / 2];
        let zero_codes = vec![0; ROTATION_BLOCK / 2];
        let extreme_stages = [
            ScanStage::new(&maximal, &maximal_scales, &maximal_codes),
            ScanStage::new(&zero, &zero_scales, &zero_codes),
        ];
        let scalar_zero = 0.0f32.to_le_bytes();
        let extreme = ScanSegment::new(
            [0xee; 32],
            1,
            ROTATION_BLOCK,
            ROTATION_BLOCK / QUANT_BLOCK,
            ROTATION_BLOCK / 2,
            extreme_stages,
            &scalar_zero,
            &scalar_zero,
        )
        .unwrap();
        let extreme_scanner = CudaUpperScanner::new(&[extreme]).unwrap();
        let coordinates = vec![1.0; ROTATION_BLOCK];
        let query = ScanQuery {
            coordinates: &coordinates,
            norm_bound: outward_norm(&coordinates).unwrap(),
        };
        let mut upper = [f64::NAN];
        extreme_scanner
            .scan_upper(query, &[extreme], &mut upper)
            .unwrap();
        assert_eq!(upper, [f64::INFINITY]);

        let ptx = emitted_f32_ptx(&cache_root);
        ptx_contract(&ptx).unwrap_or_else(|error| panic!("PTX contract failed: {error}\n{ptx}"));
        fs::remove_dir_all(&cache_root).expect("remove isolated CubeCL cache");
    }

    #[test]
    #[ignore = "requires an NVIDIA CUDA device"]
    fn cuda_raw_uppers_preserve_exact_cosine_certificate() {
        const DIMENSION: usize = 37;
        let sources: Vec<_> = (0..64).map(|row| vector(row, DIMENSION)).collect();
        let rows: Vec<_> = sources
            .iter()
            .map(|source| QuantizedRow::quantize(source, DIMENSION).unwrap())
            .collect();
        let mut globals = [Vec::new(), Vec::new()];
        let mut scales = [Vec::new(), Vec::new()];
        let mut codes = [Vec::new(), Vec::new()];
        let stages = stage_planes(&rows, &mut globals, &mut scales, &mut codes);
        let mut norms = Vec::with_capacity(rows.len() * FLOAT_BYTES);
        let mut errors = Vec::with_capacity(rows.len() * FLOAT_BYTES);
        for row in &rows {
            norms.extend_from_slice(row.reconstruction_norm_bytes());
            errors.extend_from_slice(row.error_bound_bytes());
        }
        let segment = ScanSegment::new(
            [0x42; 32],
            rows.len(),
            DIMENSION,
            rows[0].stages()[0].block_scales().len(),
            rows[0].stages()[0].codes().len(),
            stages,
            &norms,
            &errors,
        )
        .unwrap();
        let device = cubecl::cuda::CudaDevice::default();
        let f32_scanner = CudaUpperScanner::new(&[segment]).unwrap();
        let f64_scanner = CudaF64UpperScanner::new(&[segment], &device).unwrap();
        let scanners: [&dyn UpperScanner<Error = CudaError>; 2] = [&f32_scanner, &f64_scanner];
        for query_index in 0..8 {
            let query_source = vector(100 + query_index, DIMENSION);
            let query = PreparedQuery::new(&query_source, DIMENSION).unwrap();
            let certificate = CandidateCertificate::new(&query);
            for scanner in scanners {
                let mut raw_uppers = vec![0.0; rows.len()];
                scanner
                    .scan_upper(query.scan_query(), &[segment], &mut raw_uppers)
                    .unwrap();
                for (row, ((source, payload), &raw_upper)) in
                    sources.iter().zip(&rows).zip(&raw_uppers).enumerate()
                {
                    let raw_center = raw_dot_f64(query.scan_coordinates(), segment, row);
                    assert!(raw_center <= raw_upper);
                    let upper = certificate
                        .certify_upper(payload.certificate(), raw_upper)
                        .unwrap();
                    let exact = exact_cosine(&query_source, source).unwrap();
                    assert!(exact <= upper, "row {row}: exact {exact} > upper {upper}");
                }
            }
        }
    }
}
