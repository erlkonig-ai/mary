//! An `F16` (IEEE-754 half) array element for the pile.
//!
//! Defined *downstream* of triblespace via its extensible `ArrayElement` trait —
//! no triblespace change needed. Lets model weights persist at their native
//! 16-bit width (half the `F32Array` footprint). This is also the prerequisite
//! for zero-copy mmap→GPU loading: the bytes in the pile must already be in the
//! GPU tensor's dtype (f16), so there is nothing to convert and the GPU buffer
//! can alias the mmap'd pages directly.

use triblespace::core::metadata;
use triblespace::prelude::blobencodings::{Array, ArrayElement};
use triblespace::prelude::{entity, id_hex, ExclusiveId, Fragment, MetaDescribe};

/// Zero-sized marker: an IEEE-754 half-precision element (`half::f16`, 2 bytes).
pub struct F16;

impl MetaDescribe for F16 {
    fn describe() -> Fragment {
        // Fixed-id schema — minted 2026-06-23 via `trible genid`.
        let id = id_hex!("C0C8BF7450877CDC0497AD4E9463DD27");
        entity! { ExclusiveId::force_ref(&id) @
            metadata::name: "F16",
            metadata::description: "IEEE 754 half-precision float (2 bytes), native byte order",
        }
    }
}

impl ArrayElement for F16 {
    type Native = half::f16;
}

/// A flat array of `half::f16` values — the half-width counterpart to `F32Array`.
pub type F16Array = Array<F16>;
