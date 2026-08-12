#!/usr/bin/env python3
"""NVFP4 *activation* quantisation, as Inkling's `hf_quant_config.json` specifies it.

The checkpoint's `modelopt_quant_config.quant_cfg` enables an `*input_quantizer`
with exactly the same recipe as the weight quantiser -- `num_bits [2,1]` (E2M1),
`block_sizes {-1: 16, type: dynamic, scale_bits: [4,3]}` (E4M3 scales over 16
contiguous elements of the reduction axis). So activations are 4-bit too, and a
reference that runs the layer in float32 is not running the checkpoint's numerics.

TWO LEVELS, and the checkpoint says so
--------------------------------------
The scheme is two-level, and the second level is not an inference from the
config -- it is shipped. Every quantised expert tensor carries a sibling
`…​.input_amax` (BF16, shape [1]) next to the weights' `.scale2`. That amax is
the per-tensor global scale for ACTIVATIONS, calibrated offline; the per-block
E4M3 scales are what `type: "dynamic"` computes at runtime.

  s2          = input_amax / (6 * 448)            # per tensor, from the checkpoint
  block_scale = round_e4m3(block_amax / 6 / s2)   # per 16 elements, at runtime
  q           = round_e2m1(x / (block_scale*s2))  # the 4-bit code
  x_hat       = q * block_scale * s2

Dividing by `s2` before the E4M3 rounding is the entire point of the second
level: it maps a block whose amax equals the tensor amax onto E4M3's *largest*
value (448) rather than onto whatever `block_amax/6` happens to be, so the 8-bit
scale spends its 3 mantissa bits resolving the block, not the tensor's
magnitude. Drop it and the block scale is quantised at ~6% relative error, which
lands straight on the dequantised value.

Both conventions in the wild agree numerically; they differ only in which way
the global factor is stored.

  * ModelOpt / this checkpoint: `scale2 = amax / (6*448)`, a MULTIPLIER at dequant.
  * compressed_tensors: `global_scale = 6*448 / amax`, its reciprocal, and the
    effective scale is `stored_scale / global_scale`.

Authorities followed, rather than re-derived:

  * E2M1 rounding: `compressed_tensors.quantization.utils.fp4_utils.cast_to_fp4`
    (round-half-to-EVEN on the grid {0,.5,1,1.5,2,3,4,6}; the alternating
    `>` / `>=` in that function is what makes 0.25->0, 0.75->1.0, 2.5->2.0).
  * E4M3 rounding: clamp to +-448 then cast to `torch.float8_e4m3fn`, which is
    what `round_to_quantized_type_dtype` does.
  * Global scale: `generate_gparam` -> `448 * 6 / amax`.
"""
import torch

# Largest magnitude of each format. Their product is the constant that ties the
# two levels together.
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
GROUP = 16

# The E2M1 grid, low-nibble-first code order, matching the checkpoint packing.
FP4_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value, ties to even.

    Transcribed from `compressed_tensors...fp4_utils.cast_to_fp4`'s CPU path.
    Written out rather than imported so this file states the rule it depends on;
    `selftest()` checks the transcription against the library on random data.
    """
    sign = torch.sign(x)
    a = torch.abs(x)
    out = torch.zeros_like(a)
    out = torch.where((a > 0.25) & (a < 0.75), torch.full_like(a, 0.5), out)
    out = torch.where((a >= 0.75) & (a <= 1.25), torch.full_like(a, 1.0), out)
    out = torch.where((a > 1.25) & (a < 1.75), torch.full_like(a, 1.5), out)
    out = torch.where((a >= 1.75) & (a <= 2.5), torch.full_like(a, 2.0), out)
    out = torch.where((a > 2.5) & (a < 3.5), torch.full_like(a, 3.0), out)
    out = torch.where((a >= 3.5) & (a <= 5.0), torch.full_like(a, 4.0), out)
    out = torch.where(a > 5.0, torch.full_like(a, 6.0), out)
    return out * sign


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Clamp into E4M3's finite range and round to it, returning float32."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.clamp(x, finfo.min, finfo.max).to(torch.float8_e4m3fn).float()


def global_scale_from_amax(amax) -> float:
    """The checkpoint's `scale2` convention: a multiplier applied at dequant."""
    return float(amax) / (FP4_E2M1_MAX * FP8_E4M3_MAX)


def quantize_nvfp4(x: torch.Tensor, s2: float, group: int = GROUP,
                   two_level: bool = True, gparam: float | None = None):
    """Fake-quantise `x` to NVFP4 over blocks of `group` along the LAST axis.

    `s2` is the per-tensor global scale (`input_amax / (6*448)`).

    `two_level=False` reproduces the single-level variant -- the E4M3 block scale
    holds `block_amax/6` directly, with no global factor -- so the cost of
    omitting the second level can be measured rather than argued about.

    Returns `(x_hat, block_scale_e4m3, codes)`:
      * `x_hat`  dequantised float32, same shape as `x`
      * `block_scale_e4m3` float32 values of the E4M3-rounded block scales,
        shape `x.shape[:-1] + (x.shape[-1]//group,)`
      * `codes`  uint8 nibble codes into `FP4_E2M1_VALUES`, same shape as `x`
    """
    assert x.shape[-1] % group == 0, (x.shape, group)
    xb = x.reshape(*x.shape[:-1], x.shape[-1] // group, group).float()
    block_amax = xb.abs().amax(dim=-1, keepdim=True)

    if two_level:
        # Map the block's amax onto E4M3's full range before rounding it.
        block_scale = round_e4m3(block_amax / FP4_E2M1_MAX / s2)
        # `* s2` is the checkpoint's own composition (ModelOpt stores the
        # multiplier). `/ gparam` is compressed_tensors', which stores the
        # reciprocal. They disagree by one ulp of float32 -- see selftest.
        effective = block_scale * s2 if gparam is None else block_scale / gparam
    else:
        block_scale = round_e4m3(block_amax / FP4_E2M1_MAX)
        effective = block_scale

    # A zero block would divide by zero; its codes are all zero regardless.
    safe = torch.where(effective == 0, torch.ones_like(effective), effective)
    q = round_e2m1(xb / safe)
    x_hat = (q * effective).reshape(x.shape)

    # Codes, for a gate that wants to check the packing rather than the product.
    mag = q.abs()
    idx = torch.zeros_like(mag, dtype=torch.uint8)
    for i, v in enumerate(FP4_E2M1_VALUES[1:8], start=1):
        idx = torch.where(mag == v, torch.full_like(idx, i), idx)
    idx = torch.where(q < 0, idx + 8, idx)
    codes = idx.reshape(x.shape)

    return x_hat, block_scale.squeeze(-1), codes


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack a uint8 code tensor two-per-byte, low nibble first (checkpoint order)."""
    flat = codes.reshape(-1, codes.shape[-1])
    lo, hi = flat[:, 0::2], flat[:, 1::2]
    return (lo | (hi << 4)).reshape(*codes.shape[:-1], codes.shape[-1] // 2)


def selftest(seed: int = 20260812) -> None:
    """Check this file against compressed_tensors on random data.

    Two independent checks: the E2M1 rounding rule against the library's own
    `cast_to_fp4`, and the whole two-level pipeline against `fake_quantize`
    driven by the library's `generate_gparam` / `calculate_qparams`. The second
    is the one that matters -- it is the only way to know the SCALES agree and
    not merely the rounding.
    """
    from compressed_tensors.quantization.utils.fp4_utils import cast_to_fp4
    from compressed_tensors.quantization.utils.helpers import generate_gparam
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.quant_args import (
        QuantizationArgs, QuantizationStrategy, QuantizationType,
    )

    torch.manual_seed(seed)

    # 1. the E2M1 grid, including every midpoint, where ties-to-even shows up
    probe = torch.cat([
        torch.linspace(-7.0, 7.0, 20001),
        torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
                      -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0]),
    ])
    mine, theirs = round_e2m1(probe.clone()), cast_to_fp4(probe.clone())
    assert torch.equal(mine, theirs), (
        "E2M1 rounding disagrees with compressed_tensors on %d/%d values"
        % (int((mine != theirs).sum()), probe.numel()))

    # 2. the full two-level pipeline
    x = torch.randn(37, 512) * 0.07
    x[3, 17] = 2.5          # an outlier, so the global scale has work to do
    amax = x.abs().max()

    # scale_dtype is what makes the library round the block scale to E4M3 --
    # `scale_bits: [4,3]` in the checkpoint's quant_cfg. Without it the library
    # keeps float32 scales and the comparison would be against a scheme nobody
    # ships.
    args = QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT, symmetric=True,
        strategy=QuantizationStrategy.TENSOR_GROUP, group_size=GROUP,
        scale_dtype=torch.float8_e4m3fn,
    )
    gparam = generate_gparam(x.min(), x.max())
    # The library's own scale computation, per block, then its own QDQ.
    xb = x.reshape(x.shape[0], -1, GROUP)
    bmin, bmax = xb.amin(-1), xb.amax(-1)
    from compressed_tensors.quantization.utils.helpers import calculate_qparams
    ct_scale, ct_zp = calculate_qparams(bmin, bmax, args, global_scale=gparam)
    theirs = fake_quantize(x, ct_scale, ct_zp, args, global_scale=gparam)

    s2 = global_scale_from_amax(amax)
    mine, block_scale, codes = quantize_nvfp4(x, s2)

    # the scales themselves, not just the product
    assert torch.equal(block_scale.float(), ct_scale.float()), (
        "block scales differ; max |d| = %.3e"
        % float((block_scale.float() - ct_scale.float()).abs().max()))

    # Under the library's own composition the agreement is EXACT, which is what
    # proves the recipe matches rather than merely lands nearby.
    same_way, _, _ = quantize_nvfp4(x, s2, gparam=float(gparam))
    assert torch.equal(same_way, theirs), (
        "dequantised values differ under the library's own composition; "
        "max |d| = %.3e" % float((same_way - theirs).abs().max()))

    # The checkpoint's `* s2` composition differs from it by float32 round-off
    # only. Reported, not hidden: a gate that demands bit-equality across the
    # two conventions is demanding something float multiply cannot give.
    ulp = float((mine - theirs).abs().max())
    rel = ulp / float(theirs.abs().max())
    assert rel < 1e-6, "the two conventions diverge by more than round-off: %.3e" % rel

    # 3. codes round-trip through the published grid
    grid = torch.tensor(FP4_E2M1_VALUES)
    recon = grid[codes.long()].reshape(x.shape).reshape(*block_scale.shape, GROUP)
    recon = (recon * (block_scale * s2).unsqueeze(-1)).reshape(x.shape)
    assert torch.equal(recon, mine), "codes do not reproduce the dequantised values"

    # 4. reciprocal conventions really are reciprocal
    assert abs(float(gparam) * s2 - 1.0) < 1e-6, (float(gparam), s2)

    print("  ckpt-vs-library composition: max %.3e abs, %.3e rel (float32 round-off)" % (ulp, rel))
    print("nvfp4_act selftest OK "
          "(E2M1 rounding, block scales, QDQ values, codes, gparam duality)")


if __name__ == "__main__":
    selftest()
