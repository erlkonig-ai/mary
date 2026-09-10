//! Portable inference cache images, not a database or a rewind checkpoint.
//!
//! Metadata names BLAKE3-addressed raw chunks supplied by the caller. Live
//! packed KV words are moved as bytes; no quantizer participates. Page layout
//! is retained, while dead rows are reconstructed as zeros. The serving layer
//! owns publication, durability, provenance, and agreement between TP ranks.

use anyhow::{Context, Result, ensure};
use burn::tensor::{DType, Int, Tensor};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::ComputeClient;
use cubecl::server::Handle;
use serde::{Deserialize, Serialize};

use super::flash::CachedAttentionPolicy;
use super::seam::{self, Bk};

pub const CACHE_FORMAT_VERSION: u32 = 1;
pub const MAX_CHUNK_BYTES: usize = 4 * 1024 * 1024;

pub type BlobSink<'a> = dyn FnMut(&[u8]) -> Result<[u8; 32]> + 'a;
pub type BlobSource<'a> = dyn FnMut([u8; 32], usize) -> Result<Vec<u8>> + 'a;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheDType {
    F32,
    Bf16,
    I32,
}

impl CacheDType {
    pub fn bytes(self) -> usize {
        match self { Self::Bf16 => 2, Self::F32 | Self::I32 => 4 }
    }

    pub(crate) fn dtype(self) -> DType {
        match self { Self::F32 => DType::F32, Self::Bf16 => DType::BF16, Self::I32 => DType::I32 }
    }

    pub(crate) fn from_dtype(dtype: DType) -> Result<Self> {
        match dtype {
            DType::F32 => Ok(Self::F32),
            DType::BF16 => Ok(Self::Bf16),
            DType::I32 => Ok(Self::I32),
            _ => anyhow::bail!("unsupported cache tensor dtype {dtype:?}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheChunk {
    pub handle: [u8; 32],
    pub bytes: usize,
}

/// Canonical little-endian row-major bytes for `row_start..row_end`.
/// `shape` is the destination capacity, not the number of serialized rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorState {
    pub shape: [usize; 2],
    pub dtype: CacheDType,
    pub row_start: usize,
    pub row_end: usize,
    pub chunks: Vec<CacheChunk>,
}

impl TensorState {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.shape[1] > 0, "cache tensor has zero width");
        ensure!(self.row_start <= self.row_end && self.row_end <= self.shape[0],
            "cache tensor live rows exceed its shape");
        let row_bytes = self.shape[1].checked_mul(self.dtype.bytes()).context("cache row overflow")?;
        ensure!(row_bytes <= MAX_CHUNK_BYTES, "cache row exceeds transfer bound");
        self.shape[0].checked_mul(row_bytes).context("cache tensor capacity overflow")?;
        let expected = (self.row_end - self.row_start).checked_mul(row_bytes)
            .context("cache payload length overflow")?;
        let mut actual = 0usize;
        for chunk in &self.chunks {
            ensure!(chunk.bytes > 0 && chunk.bytes <= MAX_CHUNK_BYTES && chunk.bytes % row_bytes == 0,
                "cache chunk is empty, oversized, or cuts a row");
            actual = actual.checked_add(chunk.bytes).context("cache chunks overflow")?;
        }
        ensure!(actual == expected, "cache chunks contain {actual} bytes, expected {expected}");
        Ok(())
    }

    pub(crate) fn expect(&self, shape: [usize; 2], dtype: CacheDType, rows: std::ops::Range<usize>) -> Result<()> {
        self.validate()?;
        ensure!(self.shape == shape && self.dtype == dtype
            && self.row_start == rows.start && self.row_end == rows.end,
            "cache tensor shape, dtype, or live interval is incompatible");
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvPageState {
    /// Dense values or packed I32 code words.
    pub values: TensorState,
    /// Packed I32 scale words, present only for NVFP4.
    pub scales: Option<TensorState>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheState {
    pub width: usize,
    pub fp4: bool,
    /// Float dtype returned by an attention read, even on the packed arm.
    pub dtype: CacheDType,
    pub head: usize,
    pub len: usize,
    pub fill: usize,
    pub reserved: usize,
    pub epoch: usize,
    pub read: usize,
    pub pages: Vec<KvPageState>,
}

impl KvCacheState {
    pub fn validate(&self, width: usize, max_live: usize, max_capacity: usize) -> Result<()> {
        use super::kvpages::{FP4_ROW_ALIGN, MAX_PAGES, PAGE};
        ensure!(self.width == width && width > 0 && self.len <= max_live,
            "KV width or retained length is incompatible");
        ensure!(matches!(self.dtype, CacheDType::F32 | CacheDType::Bf16), "KV has nonfloat read dtype");
        ensure!(!self.fp4 || width % FP4_ROW_ALIGN == 0, "packed KV width is misaligned");
        let stored = self.head.checked_add(self.len).context("KV position overflow")?;
        if self.reserved > 0 {
            ensure!(self.pages.len() == 1 && self.epoch > 0 && self.epoch <= self.reserved,
                "invalid reserved KV geometry");
            ensure!(self.head < self.epoch && self.fill == stored
                && self.read >= stored && self.read <= self.reserved,
                "invalid reserved KV head, fill, or read interval");
            ensure!(self.pages[0].values.shape[0] == self.reserved, "KV reservation disagrees with page");
        } else {
            ensure!(self.epoch == 0 && self.read == 0 && self.head < PAGE
                && self.pages.len() <= MAX_PAGES, "invalid growing KV geometry");
        }
        if self.pages.is_empty() {
            ensure!(stored == 0 && self.fill == 0 && self.reserved == 0, "empty KV has live rows");
            return Ok(());
        }
        let mut capacity = 0usize;
        let mut real = 0usize;
        for (i, page) in self.pages.iter().enumerate() {
            let rows = page.values.shape[0];
            ensure!(rows > 0, "KV page has zero capacity");
            capacity = capacity.checked_add(rows).context("KV capacity overflow")?;
            let end = if i + 1 == self.pages.len() { self.fill } else { rows };
            let start = if i == 0 { self.head } else { 0 };
            ensure!(start <= end && end <= rows, "KV page live rows are invalid");
            ensure!(self.reserved > 0 || end > 0, "empty growing KV page");
            if self.reserved == 0 && i > 0 && i + 1 < self.pages.len() {
                ensure!(rows % PAGE == 0, "settled KV page is not page-aligned");
            }
            real = real.checked_add(end).context("KV stored-row overflow")?;
            let (cols, dtype) = if self.fp4 { (width / 8, CacheDType::I32) } else { (width, self.dtype) };
            page.values.expect([rows, cols], dtype, start..end)?;
            match (&page.scales, self.fp4) {
                (Some(scales), true) => scales.expect([rows, width / 64], CacheDType::I32, start..end)?,
                (None, false) => (),
                _ => anyhow::bail!("KV scale plane disagrees with encoding"),
            }
        }
        ensure!(real == stored && capacity <= max_capacity, "KV rows exceed stored geometry or admitted capacity");
        ensure!(self.reserved > 0 || self.head < self.pages[0].values.row_end,
            "growing KV head is outside its first page");
        Ok(())
    }

    fn tensors(&self) -> impl Iterator<Item = &TensorState> {
        self.pages.iter().flat_map(|p| std::iter::once(&p.values).chain(p.scales.iter()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttnCacheState {
    pub k: KvCacheState,
    pub v: KvCacheState,
    pub k_pre: TensorState,
    pub v_pre: TensorState,
    pub base: usize,
    pub evicted: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerCacheState {
    pub attn: AttnCacheState,
    pub attn_sconv: TensorState,
    pub mlp_sconv: TensorState,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerGeometry {
    pub layer: usize,
    pub kv_width: usize,
    pub window: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheGeometry {
    pub model_identity: [u8; 32],
    pub model_root: Option<[u8; 16]>,
    pub config_identity: [u8; 32],
    pub rank: usize,
    pub world: usize,
    pub hidden: usize,
    pub kernel: usize,
    pub sliding_window: usize,
    pub vocab: usize,
    pub context_budget: usize,
    pub extend_batch: usize,
    pub prefill_budget: usize,
    pub forbidden: Vec<usize>,
    pub router: String,
    pub shared_halved: bool,
    pub kv_prealloc: Option<usize>,
    /// Explicit Session reservation, including an entire pending prefill chunk.
    /// Omitted on the existing environment/growing lanes so their serialized
    /// compatibility identity is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_local_rows: Option<usize>,
    pub kv_epoch: usize,
    pub fp4: bool,
    pub attn_bf16: bool,
    pub act_bf16: bool,
    pub resid_bf16: bool,
    pub flash: bool,
    pub flash_fp4: bool,
    /// Legacy emits no new bytes, preserving already-published cache identity.
    /// The versioned candidate also pins its different tile/reduction grouping.
    #[serde(default, skip_serializing_if = "CachedAttentionPolicy::is_legacy")]
    pub cached_attention: CachedAttentionPolicy,
    pub head_rms_native: bool,
    pub sink_down_fused: bool,
    pub dense_fake_quant: bool,
    pub layers: Vec<LayerGeometry>,
}

impl CacheGeometry {
    fn reservation_rows(&self, context: usize, window: Option<usize>) -> Result<usize> {
        match (window, self.kv_local_rows) {
            (Some(window), Some(rows)) => {
                let needed = window.checked_add(self.prefill_budget.max(self.kv_epoch))
                    .context("local KV append capacity overflow")?;
                let needed = needed.checked_next_multiple_of(self.kv_epoch)
                    .context("local KV reservation overflow")?;
                ensure!(self.kv_prealloc.is_some() && rows == needed,
                    "explicit local KV reservation does not match admitted append width");
                Ok(rows)
            }
            _ => reservation_rows(context, window, self.kv_epoch),
        }
    }

    pub fn identity(&self) -> Result<[u8; 32]> {
        Ok(*blake3::hash(&serde_json::to_vec(&(CACHE_FORMAT_VERSION, self))?).as_bytes())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCacheState {
    pub version: u32,
    pub identity: [u8; 32],
    pub geometry: CacheGeometry,
    pub position: usize,
    pub last_prediction: Option<usize>,
    pub audio_slot: Option<usize>,
    pub vision_slot: Option<usize>,
    pub layers: Vec<LayerCacheState>,
}

impl SessionCacheState {
    /// All raw payload dependencies; no payload bytes are copied or read.
    pub fn chunks(&self) -> impl Iterator<Item = &CacheChunk> {
        self.layers.iter().flat_map(|l| {
            l.attn.k.tensors().chain(l.attn.v.tensors()).chain([
                &l.attn.k_pre, &l.attn.v_pre, &l.attn_sconv, &l.mlp_sconv,
            ])
        }).flat_map(|t| t.chunks.iter())
    }

    /// Validate the complete image before fetching a blob or allocating KV.
    pub fn validate(&self, expected: &CacheGeometry) -> Result<()> {
        ensure!(self.version == CACHE_FORMAT_VERSION, "unsupported cache image version");
        ensure!(&self.geometry == expected, "cache model, config, rank, or layout is incompatible");
        ensure!(self.identity == expected.identity()?, "cache compatibility identity mismatch");
        ensure!(expected.kernel > 0 && expected.hidden > 0 && expected.context_budget > 0
            && expected.kv_epoch > 0 && expected.world > 0 && expected.rank < expected.world
            && !expected.layers.is_empty(), "invalid cache geometry");
        ensure!(self.position > 0 && self.position <= expected.context_budget
            && self.position <= u32::MAX as usize,
            "cache image must hold a nonempty representable position");
        ensure!(self.layers.len() == expected.layers.len(), "cache layer count mismatch");
        ensure!(self.last_prediction.is_some_and(|id| id < expected.vocab), "invalid cache last prediction");
        ensure!(expected.forbidden.iter().all(|&id| id < expected.vocab)
            && expected.forbidden.windows(2).all(|w| w[0] < w[1]), "invalid forbidden-token set");
        ensure!(!expected.forbidden.contains(&self.last_prediction.expect("validated above")),
            "cache last prediction is forbidden");
        ensure!(self.audio_slot.into_iter().chain(self.vision_slot).all(|id| id < expected.vocab),
            "invalid media slot token");
        let hist = expected.kernel - 1;
        let eviction_before = self.position.saturating_sub(hist.max(expected.sliding_window));
        let mut global_gaps: Option<&[(usize, usize)]> = None;
        for (layer, geometry) in self.layers.iter().zip(&expected.layers) {
            let attn = &layer.attn;
            ensure!(attn.k.len == attn.v.len && attn.k.head == attn.v.head
                && attn.k.fill == attn.v.fill && attn.k.reserved == attn.v.reserved
                && attn.k.epoch == attn.v.epoch && attn.k.read == attn.v.read,
                "K and V page geometry disagrees");
            ensure!(attn.k.pages.iter().map(|p| p.values.shape[0]).eq(
                attn.v.pages.iter().map(|p| p.values.shape[0])), "K and V page boundaries disagree");
            let live_limit = geometry.window.unwrap_or(expected.context_budget);
            let capacity = expected.reservation_rows(expected.context_budget, geometry.window)?;
            let max_capacity = capacity.checked_add(super::kvpages::PAGE)
                .context("cache capacity limit overflow")?;
            for store in [&attn.k, &attn.v] {
                store.validate(geometry.kv_width, live_limit, max_capacity)?;
                ensure!(store.fp4 == (expected.fp4 && geometry.kv_width % super::kvpages::FP4_ROW_ALIGN == 0)
                    && store.dtype == if expected.attn_bf16 { CacheDType::Bf16 } else { CacheDType::F32 },
                    "cache KV encoding does not match the execution lane");
                ensure!(expected.kv_local_rows.is_none() || store.reserved > 0,
                    "explicitly preallocated runtime cannot restore a growing KV store");
                if store.reserved > 0 {
                    let context = expected.kv_prealloc.context("cache is reserved but runtime preallocation is off")?;
                    ensure!(store.reserved == expected.reservation_rows(context, geometry.window)?
                        && store.epoch == expected.kv_epoch, "cache reservation does not match runtime");
                }
            }
            let mut previous_end = attn.base;
            let mut skipped = 0usize;
            for &(start, end) in &attn.evicted {
                ensure!(start >= attn.base && start < end && end <= eviction_before
                    && (skipped == 0 || start > previous_end), "invalid cache eviction intervals");
                skipped = skipped.checked_add(end - start).context("cache eviction overflow")?;
                previous_end = end;
            }
            let end = attn.base.checked_add(attn.k.len).and_then(|n| n.checked_add(skipped))
                .context("cache absolute position overflow")?;
            ensure!(end == self.position, "cache layer does not end at the Session position");
            match geometry.window {
                Some(window) => ensure!(attn.evicted.is_empty()
                    && attn.base == self.position.saturating_sub(window)
                    && attn.k.len == self.position.min(window), "local cache does not hold the full window"),
                None => {
                    ensure!(attn.base == 0, "global cache has a nonzero base");
                    if let Some(gaps) = global_gaps {
                        ensure!(gaps == attn.evicted.as_slice(), "global caches disagree on evicted positions");
                    } else {
                        global_gaps = Some(&attn.evicted);
                    }
                }
            }
            attn.k_pre.expect([hist, geometry.kv_width], CacheDType::F32, 0..hist)?;
            attn.v_pre.expect([hist, geometry.kv_width], CacheDType::F32, 0..hist)?;
            layer.attn_sconv.expect([hist, expected.hidden], CacheDType::F32, 0..hist)?;
            layer.mlp_sconv.expect([hist, expected.hidden], CacheDType::F32, 0..hist)?;
        }
        Ok(())
    }
}

fn reservation_rows(context: usize, window: Option<usize>, epoch: usize) -> Result<usize> {
    ensure!(epoch > 0, "zero cache reservation epoch");
    let rows = match window {
        Some(window) => window.checked_add(epoch).context("local cache capacity overflow")?,
        None => context.max(epoch),
    };
    rows.checked_add(epoch - 1).map(|n| n / epoch * epoch).context("cache capacity rounding overflow")
}

fn transfer_rows(client: &ComputeClient<CudaRuntime>, row_bytes: usize) -> Result<usize> {
    ensure!(row_bytes > 0 && row_bytes <= MAX_CHUNK_BYTES, "invalid cache transfer row width");
    let alignment = (client.properties().memory.alignment as usize).max(1);
    let pitched_row = row_bytes.checked_add(alignment - 1).context("cache pitch overflow")?
        / alignment * alignment;
    Ok((MAX_CHUNK_BYTES / pitched_row).max(1))
}

fn export_handle(
    client: &ComputeClient<CudaRuntime>, handle: Handle, shape: [usize; 2], dtype: CacheDType,
    rows: std::ops::Range<usize>, sink: &mut BlobSink<'_>,
) -> Result<TensorState> {
    ensure!(cfg!(target_endian = "little"), "cache transport requires a little-endian device host");
    let row_bytes = shape[1].checked_mul(dtype.bytes()).context("cache row overflow")?;
    ensure!(row_bytes > 0 && row_bytes <= MAX_CHUNK_BYTES && rows.start <= rows.end && rows.end <= shape[0],
        "invalid cache export shape or interval");
    let bytes = shape[0].checked_mul(row_bytes).context("cache shape overflow")?;
    ensure!(bytes as u64 <= handle.size_in_used(), "cache tensor handle is shorter than its shape");
    let mut state = TensorState { shape, dtype, row_start: rows.start, row_end: rows.end, chunks: Vec::new() };
    // Identical boundaries for contiguous and pitched tensors: relocating a
    // page must not change its content chunks just because its strides changed.
    let rows_per_chunk = transfer_rows(client, row_bytes)?;
    for start in (rows.start..rows.end).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(rows.end);
        let offset = start * row_bytes;
        let length = (end - start) * row_bytes;
        let slice = handle.clone().offset_start(offset as u64)
            .offset_end(handle.size_in_used() - (offset + length) as u64);
        let raw = client.read_one(slice).map_err(|e| anyhow::anyhow!("cache chunk readback: {e:?}"))?;
        ensure!(raw.len() == length, "cache readback length mismatch");
        let handle = sink(raw.as_ref())?;
        ensure!(handle == *blake3::hash(raw.as_ref()).as_bytes(), "cache sink returned a non-BLAKE3 content handle");
        state.chunks.push(CacheChunk { handle, bytes: length });
    }
    state.validate()?;
    Ok(state)
}

/// Burn may pitch rows or leave a strided convolution-history view. Compact
/// only one bounded row interval at a time, never the whole page. Slice and
/// contiguous are same-dtype copies, including the I32 packed-code/scale arm.
fn export_tensor(raw: burn_cubecl::tensor::CubeTensor<CudaRuntime>, shape: [usize; 2],
    rows: std::ops::Range<usize>, sink: &mut BlobSink<'_>) -> Result<TensorState>
{
    let dtype = CacheDType::from_dtype(raw.dtype)?;
    if raw.is_contiguous() {
        return export_handle(&raw.client, raw.handle, shape, dtype, rows, sink);
    }
    let row_bytes = shape[1].checked_mul(dtype.bytes()).context("cache row overflow")?;
    ensure!(row_bytes > 0 && row_bytes <= MAX_CHUNK_BYTES && rows.start <= rows.end && rows.end <= shape[0],
        "invalid strided cache export shape or interval");
    shape[0].checked_mul(row_bytes).context("cache tensor capacity overflow")?;
    // A slice may itself need an optimized/pitched temporary before packing.
    // Budget its aligned rows too, especially narrow packed scale planes.
    let rows_per_chunk = transfer_rows(&raw.client, row_bytes)?;
    let mut state = TensorState { shape, dtype, row_start: rows.start, row_end: rows.end, chunks: Vec::new() };
    for start in (rows.start..rows.end).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(rows.end);
        let part = burn_cubecl::kernel::slice(raw.clone(), &[start..end, 0..shape[1]]);
        let part = burn_cubecl::kernel::into_contiguous(part);
        ensure!(part.is_contiguous() && part.dtype == raw.dtype,
            "cache row packing changed layout or dtype unexpectedly");
        let packed = export_handle(&part.client, part.handle, [end - start, shape[1]], dtype, 0..end - start, sink)?;
        state.chunks.extend(packed.chunks);
        // read_one above completed this copy before the next bounded slice.
    }
    state.validate()?;
    Ok(state)
}

pub(crate) fn export_float(tensor: &Tensor<Bk, 2>, rows: std::ops::Range<usize>, sink: &mut BlobSink<'_>) -> Result<TensorState> {
    let burn::tensor::TensorPrimitive::Float(raw) = tensor.clone().into_primitive() else {
        anyhow::bail!("quantized Burn float is not a cache transport tensor")
    };
    export_tensor(raw, tensor.dims(), rows, sink)
}

pub(crate) fn export_int(tensor: &Tensor<Bk, 2, Int>, rows: std::ops::Range<usize>, sink: &mut BlobSink<'_>) -> Result<TensorState> {
    let raw = tensor.clone().into_primitive();
    ensure!(raw.dtype == DType::I32, "packed cache plane is not I32");
    export_tensor(raw, tensor.dims(), rows, sink)
}

pub(crate) fn restore_float(state: &TensorState, client: &ComputeClient<CudaRuntime>, dev: &burn::backend::cuda::CudaDevice, source: &mut BlobSource<'_>) -> Result<Tensor<Bk, 2>> {
    ensure!(cfg!(target_endian = "little"), "cache transport requires a little-endian device host");
    state.validate()?;
    ensure!(state.dtype != CacheDType::I32, "integer plane used as float cache");
    let mut tensor = Tensor::<Bk, 2>::zeros(state.shape, (dev, state.dtype.dtype()));
    let row_bytes = state.shape[1] * state.dtype.bytes();
    let mut row = state.row_start;
    for chunk in &state.chunks {
        let raw = source(chunk.handle, chunk.bytes)?;
        ensure!(raw.len() == chunk.bytes, "cache blob length mismatch");
        ensure!(*blake3::hash(&raw).as_bytes() == chunk.handle, "cache blob content hash mismatch");
        let rows = chunk.bytes / row_bytes;
        let part = seam::tensor_of_dt(client.clone(), dev.clone(), client.create_from_slice(&raw), rows, state.shape[1], state.dtype.dtype());
        tensor = tensor.slice_assign([row..row + rows, 0..state.shape[1]], part);
        // Complete each upload before asking for the next blob: otherwise
        // queued staging buffers can grow into a whole-cache host copy.
        cubecl::future::block_on(client.sync()).map_err(|e| anyhow::anyhow!("cache upload sync: {e:?}"))?;
        row += rows;
    }
    Ok(tensor)
}

pub(crate) fn restore_int(state: &TensorState, client: &ComputeClient<CudaRuntime>, dev: &burn::backend::cuda::CudaDevice, source: &mut BlobSource<'_>) -> Result<Tensor<Bk, 2, Int>> {
    ensure!(cfg!(target_endian = "little"), "cache transport requires a little-endian device host");
    state.validate()?;
    ensure!(state.dtype == CacheDType::I32, "packed cache plane is not I32");
    let mut tensor = Tensor::<Bk, 2, Int>::zeros(state.shape, (dev, DType::I32));
    let row_bytes = state.shape[1] * 4;
    let mut row = state.row_start;
    for chunk in &state.chunks {
        let raw = source(chunk.handle, chunk.bytes)?;
        ensure!(raw.len() == chunk.bytes, "cache blob length mismatch");
        ensure!(*blake3::hash(&raw).as_bytes() == chunk.handle, "cache blob content hash mismatch");
        let rows = chunk.bytes / row_bytes;
        let part = seam::int_tensor_of(client.clone(), dev.clone(), client.create_from_slice(&raw), rows, state.shape[1]);
        tensor = tensor.slice_assign([row..row + rows, 0..state.shape[1]], part);
        cubecl::future::block_on(client.sync()).map_err(|e| anyhow::anyhow!("packed cache upload sync: {e:?}"))?;
        row += rows;
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Metadata only: even the large-capacity negative cases allocate no tensor
    // or payload. Fake handles are deliberate; payload verification is separate.
    fn tensor(shape: [usize; 2], dtype: CacheDType, start: usize, end: usize) -> TensorState {
        let bytes = (end - start) * shape[1] * dtype.bytes();
        TensorState { shape, dtype, row_start: start, row_end: end,
            chunks: if bytes == 0 { Vec::new() } else { vec![CacheChunk { handle: [5; 32], bytes }] } }
    }

    fn store(head: usize, len: usize) -> KvCacheState {
        KvCacheState {
            width: 64, fp4: true, dtype: CacheDType::Bf16,
            head, len, fill: head + len, reserved: 0, epoch: 0, read: 0,
            pages: vec![KvPageState {
                values: tensor([128, 8], CacheDType::I32, head, head + len),
                scales: Some(tensor([128, 1], CacheDType::I32, head, head + len)),
            }],
        }
    }

    fn fixture() -> SessionCacheState {
        let geometry = CacheGeometry {
            model_identity: [1; 32], model_root: Some([2; 16]), config_identity: [3; 32],
            rank: 0, world: 2, hidden: 128, kernel: 4, sliding_window: 4, vocab: 32,
            context_budget: 1024, extend_batch: 16, prefill_budget: 16, forbidden: vec![0, 1],
            router: "bf16".to_owned(), shared_halved: true, kv_prealloc: None, kv_local_rows: None, kv_epoch: 512,
            fp4: true, attn_bf16: true, act_bf16: true, resid_bf16: true, flash: true,
            flash_fp4: true, head_rms_native: false, sink_down_fused: false, dense_fake_quant: false,
            cached_attention: CachedAttentionPolicy::Legacy,
            layers: vec![
                LayerGeometry { layer: 0, kv_width: 64, window: None },
                LayerGeometry { layer: 1, kv_width: 64, window: Some(4) },
            ],
        };
        let layers = [None, Some(4)].into_iter().map(|window| {
            let base = window.map_or(0, |w| 12usize.saturating_sub(w));
            LayerCacheState {
                attn: AttnCacheState {
                    k: store(base, 12 - base), v: store(base, 12 - base), base, evicted: Vec::new(),
                    k_pre: tensor([3, 64], CacheDType::F32, 0, 3),
                    v_pre: tensor([3, 64], CacheDType::F32, 0, 3),
                },
                attn_sconv: tensor([3, 128], CacheDType::F32, 0, 3),
                mlp_sconv: tensor([3, 128], CacheDType::F32, 0, 3),
            }
        }).collect();
        SessionCacheState {
            version: CACHE_FORMAT_VERSION, identity: geometry.identity().unwrap(), geometry,
            position: 12, last_prediction: Some(7), audio_slot: None, vision_slot: None, layers,
        }
    }

    #[test]
    fn portable_cache_metadata_roundtrip_preserves_full_layer_state() {
        let state = fixture();
        state.validate(&state.geometry).unwrap();
        assert_eq!(state.chunks().count(), 16);
        let restored: SessionCacheState = serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
        assert_eq!(restored, state);
        restored.validate(&state.geometry).unwrap();
    }

    #[test]
    fn cached_attention_legacy_preserves_serialized_geometry_and_identity() {
        let geometry = fixture().geometry;
        // The old field order and values, before cached_attention existed.
        // Array encoding is independent of CacheGeometry's derived serializer.
        let legacy_json = format!(concat!(
            "{{\"model_identity\":{},\"model_root\":{},\"config_identity\":{},",
            "\"rank\":0,\"world\":2,\"hidden\":128,\"kernel\":4,\"sliding_window\":4,\"vocab\":32,",
            "\"context_budget\":1024,\"extend_batch\":16,\"prefill_budget\":16,\"forbidden\":[0,1],",
            "\"router\":\"bf16\",\"shared_halved\":true,\"kv_prealloc\":null,\"kv_epoch\":512,",
            "\"fp4\":true,\"attn_bf16\":true,\"act_bf16\":true,\"resid_bf16\":true,",
            "\"flash\":true,\"flash_fp4\":true,\"head_rms_native\":false,",
            "\"sink_down_fused\":false,\"dense_fake_quant\":false,",
            "\"layers\":[{{\"layer\":0,\"kv_width\":64,\"window\":null}},",
            "{{\"layer\":1,\"kv_width\":64,\"window\":4}}]}}"
        ), serde_json::to_string(&[1u8; 32]).unwrap(),
            serde_json::to_string(&[2u8; 16]).unwrap(),
            serde_json::to_string(&[3u8; 32]).unwrap());
        assert_eq!(serde_json::to_string(&geometry).unwrap(), legacy_json);
        assert_eq!(geometry.identity().unwrap(),
            *blake3::hash(format!("[1,{legacy_json}]").as_bytes()).as_bytes());
        let restored: CacheGeometry = serde_json::from_str(&legacy_json).unwrap();
        assert_eq!(restored.cached_attention, CachedAttentionPolicy::Legacy);
        assert_eq!(restored, geometry);
    }

    #[test]
    fn cached_attention_candidate_is_versioned_and_cannot_restore_as_legacy() {
        let state = fixture();
        let mut candidate = state.geometry.clone();
        candidate.cached_attention = CachedAttentionPolicy::PackedBatchedV1;
        let json = serde_json::to_string(&candidate).unwrap();
        assert!(json.contains("\"cached_attention\":\"packed-batched-v1\""));
        let restored: CacheGeometry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, candidate);
        assert_ne!(candidate.identity().unwrap(), state.identity);
        assert!(state.validate(&candidate).is_err());
        let mut candidate_state = state.clone();
        candidate_state.geometry = candidate;
        candidate_state.identity = candidate_state.geometry.identity().unwrap();
        candidate_state.validate(&candidate_state.geometry).unwrap();
        assert!(candidate_state.validate(&state.geometry).is_err());
        assert!(serde_json::from_str::<CacheGeometry>(
            &json.replace("packed-batched-v1", "packed-batched-v2")).is_err());
    }

    #[test]
    fn portable_cache_identity_names_version_model_rank_numerics_and_prefix_chunking() {
        let state = fixture();
        let identity = state.geometry.identity().unwrap();
        assert_ne!(identity, *blake3::hash(&serde_json::to_vec(&state.geometry).unwrap()).as_bytes());
        let changes: [fn(&mut CacheGeometry); 9] = [
            |g| g.model_identity[0] ^= 1,
            |g| g.config_identity[0] ^= 1,
            |g| g.rank = 1,
            |g| g.forbidden.push(2),
            |g| g.extend_batch /= 2,
            |g| g.kv_epoch /= 2,
            |g| g.flash_fp4 = !g.flash_fp4,
            |g| g.cached_attention = CachedAttentionPolicy::PackedBatchedV1,
            |g| g.router = "pre".to_owned(),
        ];
        for change in changes {
            let mut other = state.geometry.clone();
            change(&mut other);
            assert_ne!(other.identity().unwrap(), identity);
            assert!(state.validate(&other).is_err());
        }
    }

    #[test]
    fn portable_cache_metadata_rejects_malformed_shape_dtype_ranges_and_budget() {
        let state = fixture();
        let changes: [fn(&mut SessionCacheState); 13] = [
            |s| s.version += 1,
            |s| s.identity[0] ^= 1,
            |s| s.position = s.geometry.context_budget + 1,
            |s| s.last_prediction = Some(0),
            |s| s.audio_slot = Some(s.geometry.vocab),
            |s| s.layers.pop().map(|_| ()).unwrap(),
            |s| s.layers[0].attn.k.pages[0].values.dtype = CacheDType::Bf16,
            |s| s.layers[0].attn.k.pages[0].scales = None,
            |s| s.layers[0].attn.v.head += 1,
            |s| s.layers[1].attn.base -= 1,
            |s| s.layers[0].mlp_sconv.shape[1] += 1,
            |s| s.layers[0].attn.k.pages[0].values.shape[0] = usize::MAX,
            |s| s.layers[0].attn.k.pages[0].values.chunks[0].bytes = MAX_CHUNK_BYTES + 1,
        ];
        for (i, change) in changes.into_iter().enumerate() {
            let mut bad = state.clone();
            change(&mut bad);
            assert!(bad.validate(&state.geometry).is_err(), "malformed case {i} was accepted");
        }
    }

    #[test]
    fn portable_cache_accepts_old_global_gaps_but_not_incomplete_local_windows() {
        let mut state = fixture();
        state.layers[0].attn.k = store(0, 10);
        state.layers[0].attn.v = store(0, 10);
        state.layers[0].attn.evicted = vec![(2, 4)];
        state.validate(&state.geometry).unwrap();
        let mut bad = state.clone();
        bad.layers[0].attn.evicted = vec![(10, 12)];
        assert!(bad.validate(&state.geometry).is_err());
        let mut bad = state.clone();
        bad.layers[1].attn.k = store(9, 3);
        bad.layers[1].attn.v = store(9, 3);
        bad.layers[1].attn.base = 9;
        assert!(bad.validate(&state.geometry).is_err());
    }

    #[test]
    fn portable_cache_preserves_reserved_heads_fill_and_read_epoch() {
        let mut state = fixture();
        state.geometry.kv_prealloc = Some(1024);
        state.identity = state.geometry.identity().unwrap();
        for layer in &mut state.layers {
            for store in [&mut layer.attn.k, &mut layer.attn.v] {
                store.reserved = 1024;
                store.epoch = 512;
                store.read = 512;
                store.pages[0].values.shape[0] = 1024;
                store.pages[0].scales.as_mut().unwrap().shape[0] = 1024;
            }
        }
        state.validate(&state.geometry).unwrap();
        let mut bad = state.clone();
        bad.layers[1].attn.k.read = 2;
        bad.layers[1].attn.v.read = 2;
        assert!(bad.validate(&state.geometry).is_err());
    }

    #[test]
    fn portable_cache_tensor_chunks_are_bounded_whole_rows() {
        let mut state = tensor([4, 2], CacheDType::F32, 1, 3);
        state.validate().unwrap();
        state.chunks[0].bytes -= 1;
        assert!(state.validate().is_err());
        state.chunks[0].bytes = 8;
        assert!(state.validate().is_err());
        state.row_end = 5;
        assert!(state.validate().is_err());
        let empty_history = tensor([0, 64], CacheDType::F32, 0, 0);
        empty_history.validate().unwrap();
    }

    #[test]
    fn portable_cache_explicit_reservation_pins_wide_local_capacity() {
        let mut state = fixture();
        let legacy_json = serde_json::to_string(&state.geometry).unwrap();
        assert!(!legacy_json.contains("kv_local_rows"));
        let legacy: CacheGeometry = serde_json::from_str(&legacy_json).unwrap();
        assert_eq!(legacy.identity().unwrap(), state.geometry.identity().unwrap());
        state.geometry.prefill_budget = 1024;
        state.geometry.kv_prealloc = Some(1024);
        state.geometry.kv_local_rows = Some(1536);
        state.identity = state.geometry.identity().unwrap();
        assert!(state.validate(&state.geometry).is_err(), "advertised reservation must exist");
        for (layer, geometry) in state.layers.iter_mut().zip(&state.geometry.layers) {
            let rows = if geometry.window.is_some() { 1536 } else { 1024 };
            for store in [&mut layer.attn.k, &mut layer.attn.v] {
                store.reserved = rows;
                store.epoch = 512;
                store.read = 512;
                store.pages[0].values.shape[0] = rows;
                store.pages[0].scales.as_mut().unwrap().shape[0] = rows;
            }
        }
        state.validate(&state.geometry).unwrap();
        let mut wrong = state.geometry.clone();
        wrong.kv_local_rows = Some(1024);
        assert_ne!(wrong.identity().unwrap(), state.identity);
        assert!(state.validate(&wrong).is_err());
    }
}
