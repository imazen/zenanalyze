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

### The fork (needs a call)

Closing axis 3 means choosing the analysis space for the normalized path:

1. **Linear-space (extend the current design).** Port each luma/chroma content
   tier to compute on the normalized-linear plane. Honors the user's earlier
   "diffuse-white-normalized **linear light**" choice; features are linear-space
   (differ from the gamma default — `linear_light_flag_changes_variance` relies
   on this). Large, per-tier, each threshold (edge 400, uniform var<25) needs a
   linear re-derivation. SDR-in-PQ ≡ SDR *in linear space*.

2. **Normalize-then-standard (envelope undo).** One pass: decode → ×diffuse-white
   anchor (linear) → re-encode sRGB → RGB8, then run the existing calibrated
   gamma tiers unchanged. All 100 content features consistent at once, low risk,
   reuses every kernel. But it makes linear-light a NO-OP for plain SDR (breaks
   `linear_light_flag_changes_variance`) and the features are gamma-space, not
   linear — a redefinition of what the flag means.

Both use a linear diffuse-white anchor (no tone curve, per "tone mapping isn't
the solution"). Option 2 is far cheaper and gives full consistency; option 1
honors the literal "linear light" semantics. This is a feature-semantics
decision, deferred to the user — not silently chosen.
