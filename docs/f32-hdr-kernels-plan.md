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

## Validated pattern (PROVEN — compiles, SDR byte-identical)

The clean port is a **`ChunkInput` trait** (in `tier1.rs`) parameterizing *only* the
per-chunk load; the `f32x8` compute body stays shared verbatim, so there is **one
implementation** (no drift). `u8` keeps garb's tuned `vpshufb` deinterleave (zero SDR
regression); `f32` is a plain gather of an already-display-scaled-linear row. `base` is
an element index for both (row stride `width*3` elements).

```rust
pub(crate) trait ChunkInput: Copy {
    fn load_chunk8<Tok: DeinterleaveRgb24Chunk8>(rows: &[Self], base: usize, token: Tok)
        -> ([f32;8],[f32;8],[f32;8]);
}   // impl for u8 (garb), impl for f32 (gather)
```

Per kernel the change is ~2 lines: `fn k<R: ChunkInput>(token, rows: &[R], …)` and the
deinterleave becomes `let (r,g,b) = R::load_chunk8(rows, base, token);`. **Done +
verified:** `stripe_block_stats_simd<R>` — `incant!` infers `R=u8` at the call site,
`golden_is_stable` + 8 math-lock tests pass byte-identical, `fetch_f32_into` foundation
in + tested.

## Remaining (the kernel port — the bulk, ~many hours)

1. **Port the rest of the kernels** with the proven pattern:
   - `accumulate_row_simd<const BT601, const FULL, const SKIN>` — add `R` *after* the
     const generics (`incant!`'s `::<true,true,true>` turbofish leaves `R` to be inferred
     from `rgb: &[R]`); two deinterleave sites (~line 1317, 1504); CHECK the scalar edge
     stencil for any direct `rgb[i] as f32` byte reads → they already widen to f32, just
     index `&[R]` (R: Into<f32>-ish — add a `to_f32()` to `ChunkInput` if needed).
   - the Laplacian SIMD pass; `tier2_chroma.rs`; `tier3.rs` (DCT/entropy/AQ/noise);
     `palette.rs`. Each: same `<R: ChunkInput>` + `R::load_chunk8`.
2. **Add `ChunkInput::fetch_row(stream, y, &mut [Self])`** (u8 → `fetch_into`, f32 →
   `fetch_f32_into`) and make each tier's `extract_*` orchestration generic over `R`
   (stripe/scratch buffers `Vec<R>`, fetch via `R::fetch_row`, scalar luma reads via a
   `ChunkInput::to_f32`).
3. **Dispatch.** `lib.rs` ~580: `run_linear_light` ⇒ build the f32-linear stream and call
   `extract_tier{1,2,3}::<f32>` / `scan_palette::<f32>`. Retire the RGB8-clamp
   `LinearNormalized` mode (or keep only as the cheap SDR-in-HDR-envelope normalizer).
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

## Status — kernels + OETF + golden DONE (2026-06-23)

- **Kernel port DONE.** `ChunkInput` generic landed across tier1 (`3c582d6a`), tier2
  (`7b672560`), tier3 (`cabf7f6a`); palette confirmed R-agnostic (uses `borrow_row`,
  no port). SDR byte-identical the whole way (golden_is_stable + 219 lib tests).
- **OETF decision DONE (`1c7ae48f`).** The linear-light path re-encodes through the
  sRGB OETF after the exposure anchor (`linear → ×anchor → OETF → ×255`), so below
  diffuse white SDR scores the same as the gamma path (round-trip precision) and
  super-white extends past 255. This resolved "hdr vs sdr scoring differences below
  diffuse white": they match below, HDR extends above.
- **Golden + hashes DONE (`1c7ae48f`).** `extract_matrix` runs both configs
  (`[false, true]`); golden re-blessed 26→52 values/feature (gamma columns
  byte-identical), all 110 feature hashes + `feature_qualified_names.tsv` updated.
- **HDR data re-extracted + verified.** `extract_hdr_size_grid` now analyzes under
  linear-light; the imazen-26 HDR grid (1216 rows) recovers `variance` +31×,
  `edge_density` +27× vs the crushed gamma extraction.

Remaining: reconcile the SDR imazen-26 parquets to the final qualified names (values
unchanged — re-header or re-extract); perf-validate the SDR u8 path (zenbench, no
`target-cpu=native`); publish 0.2.0 (gated: README + CI-green-all-platforms + GitHub
release); migrate sibling repos.
