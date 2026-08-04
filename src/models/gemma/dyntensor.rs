//! `DynTensor` — a self-describing tensor blob schema.
//!
//! One attribute kind for any rank/dtype/shape. The blob carries its own
//! rank + dtype + shape header; the caller reads a burn `TensorData` out
//! zero-copy via the anybytes → bytes::Bytes → burn_tensor::Bytes chain.
//!
//! Layout (little-endian throughout):
//! ```text
//! [magic:     u32]   "DTNS" = 0x534E5444
//! [dtype_tag: u8]    compact tag mapped to burn::DType (see DTypeTag)
//! [rank:      u8]
//! [reserved:  u8×2]  future flags / alignment
//! [dim_0..dim_{rank-1}: u32]
//! [padding to dtype alignment]
//! [data: aligned bytes]
//! ```
//!
//! The header keeps dtype as a u8 tag rather than CBOR-encoded `burn::DType`
//! so the blob is dtype-interpretable without any deserializer. Quantized
//! / exotic dtypes (`QFloat`, `Bool(U8|U32)`) are not yet supported and
//! error at encode/decode time; extending the tag set covers them later.

use burn::tensor::{AllocationProperty, BoolStore, Bytes as BurnBytes, DType, Shape, TensorData};
use bytes::Bytes as SharedBytes;
use triblespace::core::metadata::MetaDescribe;
use triblespace::prelude::*;

pub const MAGIC: u32 = 0x534E5444; // "DTNS" in LE
pub const HEADER_BASE: usize = 4 /* magic */ + 1 /* tag */ + 1 /* rank */ + 2 /* reserved */;

/// Compact, stable u8 mapping for the `DType` variants we currently support.
/// Matches the wire byte stored in a DynTensor header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DTypeTag {
    F32 = 0,
    Bf16 = 1,
    F16 = 2,
    F64 = 3,
    Flex32 = 4,
    I64 = 5,
    I32 = 6,
    I16 = 7,
    I8 = 8,
    U64 = 9,
    U32 = 10,
    U16 = 11,
    U8 = 12,
    BoolNative = 13,
}

impl DTypeTag {
    pub fn from_dtype(dt: DType) -> Result<Self, DynTensorError> {
        Ok(match dt {
            DType::F32 => Self::F32,
            DType::BF16 => Self::Bf16,
            DType::F16 => Self::F16,
            DType::F64 => Self::F64,
            DType::Flex32 => Self::Flex32,
            DType::I64 => Self::I64,
            DType::I32 => Self::I32,
            DType::I16 => Self::I16,
            DType::I8 => Self::I8,
            DType::U64 => Self::U64,
            DType::U32 => Self::U32,
            DType::U16 => Self::U16,
            DType::U8 => Self::U8,
            DType::Bool(BoolStore::Native) => Self::BoolNative,
            other => return Err(DynTensorError::UnsupportedDtype(other.name())),
        })
    }

    pub fn to_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::BF16,
            Self::F16 => DType::F16,
            Self::F64 => DType::F64,
            Self::Flex32 => DType::Flex32,
            Self::I64 => DType::I64,
            Self::I32 => DType::I32,
            Self::I16 => DType::I16,
            Self::I8 => DType::I8,
            Self::U64 => DType::U64,
            Self::U32 => DType::U32,
            Self::U16 => DType::U16,
            Self::U8 => DType::U8,
            Self::BoolNative => DType::Bool(BoolStore::Native),
        }
    }

    pub fn from_tag(t: u8) -> Result<Self, DynTensorError> {
        Ok(match t {
            0 => Self::F32,
            1 => Self::Bf16,
            2 => Self::F16,
            3 => Self::F64,
            4 => Self::Flex32,
            5 => Self::I64,
            6 => Self::I32,
            7 => Self::I16,
            8 => Self::I8,
            9 => Self::U64,
            10 => Self::U32,
            11 => Self::U16,
            12 => Self::U8,
            13 => Self::BoolNative,
            other => return Err(DynTensorError::UnknownDtypeTag(other)),
        })
    }

    pub fn alignment(self) -> usize {
        self.to_dtype().size().max(1)
    }
}

#[derive(Debug)]
pub enum DynTensorError {
    InvalidMagic(u32),
    UnsupportedDtype(&'static str),
    UnknownDtypeTag(u8),
    TruncatedHeader,
    TruncatedData { expected: usize, got: usize },
    RankMismatch { expected: usize, got: usize },
}

impl core::fmt::Display for DynTensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMagic(m) => write!(f, "invalid DynTensor magic: {m:#x}"),
            Self::UnsupportedDtype(n) => write!(f, "dtype `{n}` not supported by DynTensor"),
            Self::UnknownDtypeTag(t) => write!(f, "unknown DynTensor dtype tag: {t}"),
            Self::TruncatedHeader => write!(f, "DynTensor blob header truncated"),
            Self::TruncatedData { expected, got } => {
                write!(
                    f,
                    "DynTensor data truncated: expected {expected} bytes, got {got}"
                )
            }
            Self::RankMismatch { expected, got } => {
                write!(f, "DynTensor rank mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for DynTensorError {}

/// `DynTensor` blob schema.
pub struct DynTensor;

impl BlobEncoding for DynTensor {}

impl MetaDescribe for DynTensor {
    fn describe() -> Fragment {
        use triblespace::core::metadata;
        let id: Id = id_hex!("418541C8207197D4C1D2BEFAD0CBE6F2");
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "DynTensor",
                metadata::description:
                    "Self-describing tensor blob: magic + dtype tag + rank + shape + aligned data. \
                     Header fully determines how bytes are interpreted; no external metadata required.",
                metadata::tag: metadata::KIND_BLOB_ENCODING,
        }
    }
}

/// Encode a [`TensorData`] as a DynTensor blob.
///
/// This allocates a fresh buffer of `header + data` size. Runtime loading
/// uses [`TryFromBlob`] which is zero-copy.
impl triblespace::core::inline::Encodes<TensorData> for DynTensor {
    type Output = Blob<DynTensor>;
    fn encode(source: TensorData) -> Blob<DynTensor> {
        let TensorData {
            bytes,
            shape,
            dtype,
        } = source;
        let tag = DTypeTag::from_dtype(dtype).expect("unsupported dtype for DynTensor");
        let rank = shape.num_dims();
        assert!(rank <= u8::MAX as usize, "rank {rank} too large");
        let align = tag.alignment();
        let unaligned_header = HEADER_BASE + 4 * rank;
        let data_offset = unaligned_header.next_multiple_of(align);

        let mut out = vec![0u8; data_offset + bytes.len()];
        out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        out[4] = tag as u8;
        out[5] = rank as u8;
        // out[6..8] reserved = 0
        for (i, d) in shape.as_slice().iter().enumerate() {
            let off = HEADER_BASE + 4 * i;
            out[off..off + 4].copy_from_slice(&(*d as u32).to_le_bytes());
        }
        out[data_offset..].copy_from_slice(&bytes);
        Blob::new(anybytes::Bytes::from_source(out))
    }
}

/// Parse a DynTensor blob back into a [`TensorData`] with zero-copy bytes.
impl TryFromBlob<DynTensor> for TensorData {
    type Error = DynTensorError;

    fn try_from_blob(blob: Blob<DynTensor>) -> Result<Self, Self::Error> {
        let bytes = blob.bytes;
        let header = parse_header(&bytes)?;

        // Slice out the data region zero-copy, then bridge into burn's Bytes.
        let data = bytes.slice(header.data_offset..);
        let expected = header.byte_count;
        if data.len() < expected {
            return Err(DynTensorError::TruncatedData {
                expected,
                got: data.len(),
            });
        }
        let data = data.slice(..expected);

        let shared: SharedBytes = data.into();
        let burn_bytes = BurnBytes::from_shared(shared, AllocationProperty::File);

        Ok(TensorData {
            bytes: burn_bytes,
            shape: Shape::new_raw(header.dims.into_iter().collect()),
            dtype: header.dtype,
        })
    }
}

/// Parsed DynTensor header fields.
#[derive(Debug, Clone)]
pub struct ParsedHeader {
    pub dtype: DType,
    pub dims: Vec<usize>,
    /// Offset of the data region, in bytes, from blob start.
    pub data_offset: usize,
    /// Number of payload bytes (product of dims × element size).
    pub byte_count: usize,
}

pub fn parse_header(bytes: &anybytes::Bytes) -> Result<ParsedHeader, DynTensorError> {
    if bytes.len() < HEADER_BASE {
        return Err(DynTensorError::TruncatedHeader);
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(DynTensorError::InvalidMagic(magic));
    }
    let tag = DTypeTag::from_tag(bytes[4])?;
    let rank = bytes[5] as usize;
    let unaligned_header = HEADER_BASE + 4 * rank;
    if bytes.len() < unaligned_header {
        return Err(DynTensorError::TruncatedHeader);
    }
    let mut dims = Vec::with_capacity(rank);
    for i in 0..rank {
        let off = HEADER_BASE + 4 * i;
        let d = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        dims.push(d);
    }
    let align = tag.alignment();
    let data_offset = unaligned_header.next_multiple_of(align);
    let elems: usize = dims.iter().copied().product();
    let byte_count = elems * tag.to_dtype().size();

    Ok(ParsedHeader {
        dtype: tag.to_dtype(),
        dims,
        data_offset,
        byte_count,
    })
}
