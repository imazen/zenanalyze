# zenanalyze consistency audit — 2026-06-18

Goal: feature values consistent across **SIMD/scalar**, **channel type**, and
**SDR/HDR**, with every feature enabled (`--features experimental,hdr`).
Harnesses: `examples/consistency_matrix.rs` (channel type),
`examples/sdr_hdr_consistency.rs` (envelope). Commit context: `2705ce1`+.

## Axis 1 — SIMD vs scalar: CONSISTENT ✓ (fixed)

tier1 block-stats ran two luma definitions — the `f32x8` kernel (no floor) for
full stripes and a hand-written floored-u32 scalar tail for partial stripes —
so `uniformity`/`flat_color`/`block_var` mixed both and could flip near the
`var<25`/`range<=4` thresholds by CPU tier. Unified into one row-parameterized
magetypes kernel (`fix … kill SIMD/scalar split`, `0bc8d4b`); the SIMD and
scalar arms are now generated from one source. Also fixed a latent partial-tail
normalization bug (divided by 64 regardless of row count). Guarded by 2 new
`block_stats_tests`.

## Axis 2 — channel type (u8 / u16 / f32): CONSISTENT ✓ (fixed)

`consistency_matrix`: **108/108 content features bit-identical** across u8,
losslessly-promoted u16, and f32-sRGB. Two fixes:
- **p99/peak luminance** were split — u8+SDR hard-returned the 80-nit *display
  reference* via the depth fast path; u16/f32 walked pixels for *content*
  luminance. The first attempt (`2705ce1`) made it content-referred by REMOVING
  the fast path — but that cost ~5.7ms/4MP and broke the flat-per-tier perf, so
  it was reverted (`0595292`). Final resolution: EXTEND the fast path to every
  non-HDR transfer (u8/u16/f32) — all SDR sources short-circuit to the same
  display-referred profile (peak/p99 = 80), consistent AND fast. Only true HDR
  (PQ/HLG) walks. Display-referred for SDR is by design (the depth tier answers
  "what dynamic range does this need", not "how bright is this pixel").
- **effective_bit_depth** kept the byte-replication signature (`low==high` ⇒ 8
  for promoted u16) via a cheap byte-only probe inside the fast path.

Remaining divergences are format/precision-descriptive **by design**:
`bitmap_bytes` (byte count, 1/2/4 bpc) and `effective_bit_depth` for f32
(reports 32-bit *storage* depth, the documented contract).

## Axis 3 — SDR vs HDR envelope: CONSISTENT ✓ (fixed via option A)

`sdr_hdr_consistency`: SDR content tagged SDR vs the same content as
SDR-white-anchored PQ u16, linear-light ON for both, content features only.

- Before: **27/100** invariant — the linear path re-derived only Variance;
  every other content tier ran on the PQ→RGB8 narrowing that crushed the
  envelope toward black.
- After: **97/99** content features invariant (bitmap_bytes excluded as a format
  byte-count). The 2 residuals are higher-order statistics sensitive to the
  u8-linear quantization — `LumaKurtosis` (4%) and `SpectralSlopeY` (7%) — both
  well within "sufficiently close".

### Fix: normalize ONCE at the shared RowStream layer

The content tiers share combined passes and all read the SAME `RowStream` RGB8
(`Inner::Convert` is the single narrowing point for u16/f32/HDR). So the fix was
one new constructor — `RowStream::new_normalized_linear` — that, when
linear-light is on, decodes each row to RGBF32_LINEAR, applies the diffuse-white
anchor (a linear ×scale, **no tone curve**), and quantizes to display-range
RGB8. Every content tier then reads envelope-normalized bytes in its existing
combined pass, so SDR-in-HDR ≡ SDR for all of them at once — no per-tier port.

Per the user's choice this is **linear-space** (option A): features stay
linear-light (the flag still meaningfully changes SDR variance), 8-bit (the
tiers' existing precision). This **deleted the separate `linear_tier` pass** —
it existed only to re-derive Variance, exactly the anti-combine duplication;
`anchor_scale` moved into `row_stream`, and `linear_tier.rs` is now a test-only
integration module.

The default path (linear-light OFF) is byte-for-byte unchanged — its zero-copy
Native / StripAlpha fast paths are untouched; only `with_linear_light(true)`
pays the inherent decode→normalize cost.
