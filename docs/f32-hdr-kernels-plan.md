# f32 tier kernels — true full-range HDR-correct features

## Why

On HDR (PQ/HLG) input the default analyzer narrows to gamma RGB8, which treats the
PQ peak (10000 nits) as display-white and **crushes content to near-black** — the
imazen-26 HDR feature data measured `variance` 0.09× / `edge_density` 0.006× of the
SDR equivalent (garbage). The opt-in linear-light path (`with_linear_light`) fixes
the *displayable* range but still **re-quantizes to RGB8**, so super-white HDR
hard-clips to 255 (only the source-direct depth tier captures the envelope). The 8-bit
round-trip is the limiter: in RGB8 you must choose between display-precision (clip
highlights) and full-range (crush displayable precision). The correct fix is to run
the tier kernels on **`RGBF32_LINEAR` directly — no re-quantization** — so the full
HDR range survives at f32 precision with display-white at a reference below 1.0.

The tier kernels **already compute in f32/f64** (they widen u8 → `f32x8` at load);
the only thing that's u8 is the *input row*. So this is a load-path change, not a math
rewrite.

## Done (foundation)

- `RowStream::fetch_f32_into(y, &mut [f32])` + `normalize_linear_row_f32` — emits
  display-scaled f32 (`linear * scale`, **unclamped**), display-white = `255.0` (so
  the feature *scale* matches the u8 path), super-white survives as `> 255.0`.
- Test `fetch_f32_preserves_superwhite_where_u8_clips` (linear_tier.rs): a 2×-display-white
  PQ pixel reads `~510` via `fetch_f32_into` vs `255` (clamped) via `fetch_into`.
- Prior: `linear_light_moves_all_rowstream_tiers_not_just_variance` proved the
  linear-light row-stream already feeds every tier (it's the *RGB8 clamp* that loses
  HDR, not a Variance-only limitation).

## Remaining (the kernel port — the bulk)

1. **Generic row load over u8 / f32 (no SDR perf regression).** Keep the u8 SIMD fast
   path for the default SDR case; add an f32 path for HDR-correct. Make each kernel's
   load generic over a row source that yields `f32x8` lanes — u8 inner widens
   (`u8→f32x8`, today's path), f32 inner loads `fetch_f32_into` rows directly. The
   `f32x8`/`f64` compute body is shared verbatim, so there is **one implementation**
   (no drift — `two code paths → bug` rule). Files: `tier1.rs` (`extract_tier1_into_dispatch`,
   the `[f32;8]` load at the top), `tier2_chroma.rs`, `tier3.rs` (DCT/entropy/AQ/noise),
   `palette.rs`. `borrow_row` / `fetch_range` need f32 twins too.
2. **Dispatch.** `lib.rs` ~580: when `run_linear_light`, build the f32-linear stream and
   route the tiers through the f32 load. Retire the RGB8-clamp `LinearNormalized` mode
   (or keep it only as the cheap SDR-in-HDR-envelope normalizer — decide during the port).
3. **Anchor below 1.0.** Confirm display-white maps to `255.0` and peak floats above
   (no clamp); the f32 kernels then see the true contrast. No precision loss (f32).
4. **Golden re-bless.** The default (gamma) golden is unaffected (f32 path is opt-in).
   Add linear-light golden coverage if we want the f32-HDR values pinned cross-platform
   (watch the SIMD-reduction tolerances — same xplat discipline as the gamma golden).
5. **Perf-validate (no `target-cpu=native`).** zenbench the SDR u8 path before/after to
   prove the generic load didn't regress it; bench the f32 HDR path so its cost is known.

## After the kernels

6. **Re-extract the imazen-26 HDR data** with the f32 linear-light path (the native 76 +
   grid 1216), re-validate the regular features are no longer crushed, re-mirror to Tower,
   update `benchmarks/imazen26_features_2026-06-23.pointer.md`.
7. **Publish zenanalyze 0.2.0** (release prep already done on main: CHANGELOG reconciled,
   dry-run clean, semver-checks OK; tag `zenanalyze-v0.2.0` → GitHub release → publish),
   then resume the sibling-repo migration.

## Status

Foundation done + tested + pushed. The kernel port (step 1) is a careful SIMD load-path
refactor across tier1/2/3 + palette — a focused multi-hour effort that should be done
deliberately (perf + golden discipline), not rushed.
