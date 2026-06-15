# zenanalyze review: channel-type consistency, precision utilization, linear light, HDR capture (2026-06-14)

**Question (the spec to assess).** Does zenanalyze produce SDR feature values
that are **consistent regardless of channel data type** (u8 / u16 / f32),
**unless genuine additional precision is present and utilized**, with the math
in **linear light**, while **HDR is also captured**?

**Method.** Code review of `src/row_stream.rs` (the row source feeding tiers
1/2/3 + Palette) plus an empirical probe (`examples/channel_consistency_probe.rs`)
that feeds the *same* sRGB content five ways and diffs nine SDR features. HDR
capture assessed separately via the source-direct depth tier (the 76-row
`imazen26_hdr_features_2026-06-14` extraction).

## Verdict

| Spec requirement | Status | Evidence |
|---|---|---|
| Consistent across channel data type | ✅ **MET** | u8 vs lossless-u16 vs f32 → 9/9 identical features |
| Utilize genuine extra precision when present | ❌ **NOT MET** | sub-8-bit detail → 0/9 features react (discarded) |
| Linear-light analysis | ❌ **NOT MET** | linear-tagged vs sRGB-tagged bytes → 9/9 identical (transfer-blind) |
| HDR captured | ✅ **MET** | depth tier: peak 379–1646 nits, headroom 2.2–4.4 stops, 76/76 rows |

## Probe output (SDR tiers only — built `--features experimental`, no `hdr`)

```
feature                          A:u8    B:u16loss    C:u16+det     D:u8 lin   E:f32 srgb
Variance                  5460.941406  5460.941406  5460.941406  5460.941406  5460.941406
EdgeDensity                  0.000000     0.000000     0.000000     0.000000     0.000000
ChromaComplexity             0.005888     0.005888     0.005888     0.005888     0.005888
HighFreqEnergyRatio          0.013113     0.013113     0.013113     0.013113     0.013113
LumaHistogramEntropy         4.775449     4.775449     4.775449     4.775449     4.775449
LaplacianVariance            0.000270     0.000270     0.000270     0.000270     0.000270
AqMapMean                    2.411079     2.411079     2.411079     2.411079     2.411079
NoiseFloorY                  0.122403     0.122403     0.122403     0.122403     0.122403
(Uniformity 1.0 across all; EdgeDensity 0 — flat-ish synthetic)

A==B (u16 lossless): 9/9 identical   channel-type consistency, u16
A==E (f32 sRGB):     9/9 identical   channel-type consistency, f32
A==C (u16 +detail):  9/9 identical   0 features utilize genuine sub-8-bit precision
A==D (linear tag):   9/9 identical   transfer-BLIND: math on code values, NOT linear light
```

- **A** u8 RGB8_SRGB (baseline).
- **B** u16 lossless promotion `(c<<8)|c` = c·257 (byte replication; 255→65535).
- **C** B + genuine sub-8-bit detail, amplitude ≤ 0x3f (0.245 of an 8-bit LSB —
  never crosses a narrowing rounding boundary).
- **D** A's exact bytes re-tagged `RGB8` (linear transfer).
- **E** f32 sRGB-encoded floats `c/255.0`, `RGBF32` tagged sRGB.

**Promotion gotcha (verified, avoids a false finding):** the lossless 8→16
promotion is byte-replication `(c<<8)|c` (=c·257), **not** `c<<8` (=c·256). The
converter narrows with correct full-range rounding `round(v·255/65535)`, so a
`c<<8` promotion reads back ~0.39% low — an artifact, not an inconsistency. The
first probe run used `c<<8` and showed spurious 6/9 mismatches; with the correct
promotion it is 9/9 identical.

## Root cause

`RowStream::new` (src/row_stream.rs) routes tiers 1/2/3 + Palette through
`RowConverter::new(desc, PixelDescriptor::RGB8_SRGB)` for any non-RGB8-layout
input — i.e. **everything is narrowed to 8-bit, sRGB-gamma RGB before the SDR
feature math runs**, and the math then operates on those code values without
linearizing. Consequences, each confirmed by the probe:

1. **Consistency is real but achieved by narrowing.** u8 / u16 / f32 of the same
   content all land on the same RGB8_SRGB row, so features match exactly (A==B==E).
   This satisfies the consistency requirement robustly.
2. **Precision cannot be utilized.** Narrowing to 8-bit discards everything below
   one 8-bit LSB *before* the features see it (A==C, 0/9). The spec's "unless
   actual additional utilized precision is present" clause can never fire for
   tiers 1/2/3 — there is no path by which real >8-bit detail reaches them.
3. **Not linear light.** The Native path is layout-compatible-only and explicitly
   transfer-agnostic ("the analyzer math doesn't care about transfer"), so
   sRGB-gamma vs linear code values produce identical features (A==D). SDR
   features are computed in the *encoded* (gamma) domain, not linear light.

HDR is the exception by design: the depth tier and alpha tier read the **source
bytes directly** (bypassing the RGB8 narrowing), which is why the 16-bit PQ
renditions light up peak-nits / headroom / hdr_present correctly. The HDR signal
"would not survive RowConverter narrowing" — and doesn't have to.

## Recommendation

Consistency and HDR are already met. Closing the precision + linear-light gaps is
**one change**: compute tiers 1/2/3 on a **higher-precision, linear intermediate**
(linear f32, or linear u16) instead of RGB8_SRGB. Linearizing a lossless
promotion still collapses to the same values, so the A==B/E consistency is
preserved; genuine >8-bit detail now survives (closes #2); and the math is in
linear light (closes #3).

This is a **significant, deliberate change — not a silent one:**

- **It re-bases every SDR feature's numeric value.** Every downstream fitted
  threshold / picker / model trained on the current gamma-8-bit features would be
  miscalibrated and must be retrained. Per the crate threshold contract numeric
  drift is allowed within a minor, but this is a wholesale recalibration, not a
  tweak.
- **Perf + SIMD.** Today's hot kernels operate on RGB8; a linear-f32 path adds a
  per-pixel EOTF and widens the working set. The `linear-srgb` dep (already in
  the tree, `transfer` feature) supplies batch sRGB/PQ/HLG EOTFs to build on.
- **Not uniformly desirable per feature.** Gamma-domain analysis is a defensible,
  perceptually-motivated choice for some features (edge density, AQ map, variance
  weight shadows more in gamma) since the analyzer feeds codec decisions that
  often live in a perceptual space. True-luminance features (noise floor, peak,
  histogram entropy) are the ones that clearly *want* linear. A blanket switch
  may help some features and hurt others — worth a per-feature A/B before
  committing the default.

**Suggested path:** add it as an **opt-in** first — an `AnalysisQuery` flag (or a
`linear` cargo feature) that runs tiers 1/2/3 in linear higher-precision, leaving
the default (gamma/8-bit) untouched so existing fitted models keep working. A/B
the two domains on the imazen-26 corpus (SDR + the 16-bit HDR renditions), then
decide per-feature which domain becomes the default in the next minor. This is a
user decision (it invalidates fitted consumers) — flagged, not auto-applied.

## Reproduce

```bash
cargo run --release --features experimental --example channel_consistency_probe
```
