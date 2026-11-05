# Analyzer shared-pixel qualification (2026-11-05)

Base main `7064f1df`; pixel/converter source
`fa836abf72afe9826e866dc25712b7c800323962`. Production requires 0.3.1; the
example-only resampler uses `e3975fb9d6d6b7baa96038a0eb8e27febb37c012`.
The normal root graph has one pixel identity. Historical 0.2.14 packages exist
only in the workspace-excluded oracle, never in the production graph.

The initial cohort compiled after private descriptor-tag and moved-import
repairs, but 10 of 239 analyzer tests failed with HdrSourceRequiresPeak (225
passed, four existing ignores). Both golden/versioning gates failed at converter
construction. The approved preservation path adds private analysis stages;
existing assertions, thresholds, model files and golden files remain unchanged.

## Compatibility evidence

- `just sibling-descriptor-check`: 1,400 descriptor hashes equal the independent
  historical implementation using real 0.2.14 enum casts.
- The production HDR helper is compiled directly into the isolated old-runtime
  oracle. All 1,680 descriptor cases construct successfully: U8/U16/F16/F32,
  PQ/HLG, BT.709/BT.2020/P3/AdobeRGB/Unknown, RGB/four RGBA alpha modes/Gray/
  GrayAlpha, widths 1/7/16/31/257/65,536, two rows with padding. Nonopaque alpha
  and dense integer/half-float code ranges are included.
- 3,360 rows match exactly, comparing 110,624,640 bytes. Of these, 960 gray rows
  compare against equivalent replicated-RGB old inputs: the old HDR gray path
  panics at width one. No existing test is skipped to obtain this result.
- `just sibling-hdr-check`: unchanged analyzer gate, 235 pass, zero fail, four
  existing ignores, including feature-name and feature-value golden checks.
- `just sibling-cohort-check`: 671 runtime passes across 34 test binaries, four
  existing ignores; 21 doctest passes, nine existing ignores. The real metapicker
  bake is supplied explicitly (104,193 bytes; SHA-256
  `4479ef9c874ebf1c83d059794d2959e725c1ea9d1aa77a5721c7a26a31624e34`),
  alongside the repository's shipped predictor bakes for forward parity.
- Strict workspace/all-target/all-feature clippy and standalone fuzz-target
  compilation pass. Neither oracle nor examples run an encoder qualification.

The adapter preserves historical transfer kernels and quantizers, non-FMA
matrix arithmetic, stored-color alpha semantics, and the U8 encoded-domain
matrix placement. It is an analysis representation, not HDR rendering. It adds
no source peak, tone mapping or public API. The source-direct HDR/alpha tiers and
opt-in linear-light analysis path retain their existing behavior.

The first full cohort attempt passed runtime/docs but found the resampler's old
pixel lock in all-target compilation. After precise lock convergence the next
attempt found one moved DiffuseWhite example import. Both failed logs are kept.
The final lint/fuzz scope passes: rc0, 6s, peak RSS 0.33 GiB, minimum available
7,989 MiB, peak load 3.15. The successful expanded oracle plus analyzer scope:
rc0, 75s, peak RSS 0.59 GiB, minimum available 7,867 MiB, peak load 2.60.

Full logs, resolved locks, source archives and source identities are retained at
`/mnt/v/output/zensim-siblings-20261105/zenanalyze/` (checked on dev), with the
approved continuation under `approved-hdr/`. Mirror:
`/mnt/tower/output/zensim-siblings-20261105/zenanalyze/`.
