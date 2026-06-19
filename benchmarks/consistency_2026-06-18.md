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
losslessly-promoted u16, and f32-sRGB. Two fixes (`2705ce1`):
- **p99/peak luminance** were semantically split — u8+SDR hard-returned the
  80-nit *display reference*, u16/f32 computed the real *content* luminance.
  Field docs say content-referred; removed the shortcut. Solid gray-128 now
  reports ~17 nits on every channel type.
- **effective_bit_depth** over-reported promoted u16 as 14. Added the
  content-independent byte-replication signature (`low==high` ⇒ 8).

Remaining divergences are format/precision-descriptive **by design**:
`bitmap_bytes` (byte count, 1/2/4 bpc) and `effective_bit_depth` for f32
(reports 32-bit *storage* depth, the documented contract).

## Axis 3 — SDR vs HDR envelope: PARTIAL (27/100), architectural

`sdr_hdr_consistency`: SDR content tagged SDR vs the same content as
SDR-white-anchored PQ u16, linear-light ON for both, content features only:

> **27/100 features SDR/HDR-invariant.** 73 still collapse under the envelope.

Why: the diffuse-white-normalized linear-light path (`linear_tier`) currently
re-derives **only Variance** on normalized linear; every other content tier
runs on the PQ→RGB8 narrowing, which crushes the SDR-in-PQ envelope toward
black. The depth tier (HDR capture, ids 32-39,46,47) IS consistent and is
correctly excluded (it measures the envelope on purpose).

### Fix: normalize ONCE at the shared RowStream layer (not per-tier)

The content tiers don't have per-feature kernels — they share combined passes
(tier1's `accumulate_row_simd` + `stripe_block_stats_simd` emit variance, edges,
chroma, uniformity, covariance in ONE f32x8 sweep; tier3 fuses DCT/entropy/AQ/
noise/line-art/gradient/quant-survival). And every content tier reads the SAME
`RowStream` RGB8 (`src/row_stream.rs`, `Inner::Convert` is the single narrowing
point for u16/f32/HDR sources). So the fix is **one change at RowStream**, not a
per-tier port: when linear-light is on, the Convert path emits diffuse-white-
normalized RGB8 (decode → linear → ×anchor → re-encode), and every content
feature becomes SDR/HDR-consistent inside its existing combined pass at once.

This also **deletes `linear_tier`** — that separate luma-plane pass exists only
to re-derive Variance, which is exactly the anti-combine duplication; folding
the anchor into the shared RowStream input makes tier1 emit a consistent
Variance (and everything else) for free.

Semantic consequence to confirm: linear-light becomes "undo the HDR envelope,
then analyze in the standard display-RGB space" — a NO-OP for plain SDR (so
`linear_light_flag_changes_variance` is re-expressed as an envelope test), and
features are display-space, not linear-space. The anchor itself is still applied
in linear light (no tone curve, per "tone mapping isn't the solution").
