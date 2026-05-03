# Held-out re-encode A/B harness — template for follow-up codecs

Status: 2026-05-04. Reference implementation: `zenwebp/dev/picker_v0_3_holdout_ab.rs`
(commit `imazen/zenwebp@1f46e06`).

## What this is

A single-codec held-out A/B harness that compares picker-tuned encodes
against the bucket-table baseline on a held-out PNG corpus, at multiple
`target_zensim` levels. Output: a per-codec markdown verdict
(SHIP/HOLD) with per-target byte deltas and achieved-zensim parity
checks.

## Why one harness per codec

Codecs differ on: (1) the encoder-knob axes they expose, (2) the
in-runtime picker integration site (and which `.bin` it embeds), (3)
the closed-loop quality-search API (`target_zensim` / `target_psnr` /
target_ssim2). The harness lives in each codec's `dev/` so it can:

- Pin per-codec-specific encoder builders (`with_method` /
  `with_segments` / `with_distance` / etc.).
- Bypass the in-runtime `picker` feature gate so it can A/B against
  newer .bin versions whose `schema_hash` doesn't match the embedded
  one.

## Skeleton

```rust
#![cfg(all(feature = "target-zensim", feature = "analyzer"))]
#![forbid(unsafe_code)]

use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpredict::{AllowedMask, Model, Predictor, ScoreTransform,
                 argmin::argmin_masked_in_range};

const FEAT_COLS: &[&str] = &[ /* mirror src/encoder/picker/spec.rs::FEAT_COLS */ ];
const ANALYSIS_FEATURES: &[Option<AnalysisFeature>] = &[ /* parallel array */ ];
const RAW_TRANSFORMS: &[RawTransform] = &[ /* dump the .bin's
                                              feature_transforms entry
                                              and transcribe */ ];
const CELLS: &[/* per-codec tuple */] = &[ /* lex-sorted */ ];

fn main() {
    // 1. Parse args (--bin, --corpus, --targets, --out-md, --out-tsv).
    // 2. Read .bin via Model::from_bytes (no schema check).
    // 3. For each PNG:
    //    a. Decode RGB8.
    //    b. Extract raw features in FEAT_COLS order.
    //    c. Apply log/log1p/identity transforms in-place.
    //    d. Build engineered vec (codec-specific layout).
    //    e. Run Predictor::predict.
    //    f. Decode argmin cell + scalars → encoder knobs.
    //    g. Encode picker arm + bucket arm at each target_zensim.
    //    h. Score bytes; write TSV row.
    // 4. Aggregate; write markdown verdict.
}
```

## What changes per codec

### zenwebp (reference)

- `CELLS = [(method, segments)]` over `{4,5,6} × {1,4}` (6 cells).
- Knobs: `(method, segments, sns_strength, filter_strength,
  filter_sharpness)`.
- Bucket fallback: `classify_image_type_rgb8 + content_type_to_tuning`,
  method pinned to 4.
- Closed loop: `LossyConfig::with_target_zensim`.

### zenavif (TODO)

Adapt the harness shape from zenwebp. Key changes:

- `CELLS` definition will reflect zenavif's published cell taxonomy
  (`zenavif/src/encoder/picker/spec.rs::CELLS`). At time of writing
  v0.3 zenavif uses `(speed, tune, threads_chunks)` or similar — read
  the codec's `spec.rs` before writing the harness.
- Knobs: `EncodeRequest` builder is named differently than zenwebp's
  (`AvifEncoder::with_speed`, `with_tune`, etc.). The closed loop is
  `target_ssim2` (zenavif uses ssim2, not zensim) — read zenavif/api.rs
  for the right closed-loop method.
- Bucket fallback: zenavif's `content_type_to_tuning` lives in
  `zenavif/src/encoder/analysis/`. Confirm method/threads pin matches
  zenavif's Auto-preset code path.
- Engineered features: same `FEAT_COLS + size_oh[4] + poly[5] +
  cross[N_FEAT] + icc[1]` shape (the trainer is shared) UNLESS zenavif
  has additional codec-specific axes — check the v0.3 manifest.json's
  `extra_axes` list.

### zenjxl (TODO)

Same shape as zenavif. Key changes:

- `CELLS` reflects zenjxl's cell taxonomy (likely `(effort, distance)` or
  `(method, modular)`).
- Knobs: `JxlEncoder::with_distance`, `with_effort`, `with_chroma_subsampling`.
- Closed loop: zenjxl supports `target_butteraugli` / `target_distance`
  natively; check whether `target_zensim` is plumbed.
- Bucket fallback: zenjxl/src/encoder/analysis/.

## Brief checklist for each follow-up codec

1. Pull the v0.3+ `.bin` and `manifest.json` for the codec from R2.
2. Read `manifest.json::feat_cols` and `extra_axes` — if those don't
   match the codec's `spec.rs::FEAT_COLS` order, STOP and fix the
   trainer config or the codec's spec.rs before the A/B (running
   with mis-ordered features produces silently-wrong picks).
3. Read the .bin's `feature_transforms` metadata. If `len !=
   n_inputs`, the .bin has the same baker bug as zenwebp v0.3 — either
   patch the .bin (extend with identity for engineered axes) or strip
   the entry and apply transforms manually.
4. Inspect the .bin's `metadata::feature_bounds` for inverted/odd
   transforms (e.g. log-bounded ranges on what should be linear features).
5. Write the codec-specific harness file under `<codec>/dev/`.
6. Register it in `<codec>/Cargo.toml` with `required-features =
   ["target-<metric>", "analyzer"]`.
7. Run on the same `~/work/zentrain-corpus/mlp-validate/<codec>-val/`
   subset (or the codec's equivalent held-out split).
8. Push the markdown verdict to `imazen/zenanalyze:benchmarks/picker_v0.3_<codec>_<date>.md`.
9. Upload the markdown + TSV to `s3://zentrain/<codec>/pickers/`.
10. Reference back to this template doc and the zenwebp harness in the
    verdict file's methodology section.

## Common gotchas

- **Don't enable the codec's in-runtime `picker` feature in the harness.**
  That feature gates code that consumes the *embedded* .bin (typically
  pinned at compile time to one schema_hash). The held-out A/B is
  designed to test a NEW .bin that may not match — load it externally
  via `zenpredict::Model::from_bytes`.
- **Don't reuse the codec's existing `picker_ab_eval.rs`.** Those use
  `Preset::Auto` which routes through the embedded picker (or
  bucket-table fallback). For a held-out A/B against a newer .bin,
  bypass Auto and set every encoder knob explicitly.
- **Mind the `target_*` API differences.** zenwebp's
  `with_target_zensim` accepts `ZensimTarget::new(t).with_max_passes(n)`;
  zenavif and zenjxl have analogous but not identical builders. Read
  each codec's `target_*` test for the canonical incantation.
- **The codec's `content_type_to_tuning` may not pin method.** Read
  the Auto path in `<codec>/src/encoder/vp8/mod.rs` (or equivalent) to
  see exactly which knobs the bucket-table fallback sets vs which it
  inherits from the user's `params` struct. Mirror that pinning in
  the bucket arm of the harness.
- **Achieved-metric tolerance: 0.5pp.** If bytes_picker < bytes_bucket
  but achieved_picker is more than 0.5pp below achieved_bucket, the
  comparison is invalid (picker is "winning" by encoding worse output).
  The template harness reports both deltas; the verdict logic in
  zenwebp pulls SHIP only when the achieved-zensim gap is ≤ 0.5pp.

## Reference timings (zenwebp, AMD 7950X)

| Step | Wall (per image, q=80) |
|---|---:|
| zenanalyze 36-feature extraction | ~10–30 ms |
| Engineered-vec build + predict | <1 ms |
| Encode (closed-loop, m4, 3 passes max) | 70–150 ms |
| Total per (image, target) — both arms | ~200–400 ms |

41 images × 4 targets × 2 arms ≈ 33 s on cid22-val. Plan for ~1–3 min
per codec with `--max-passes 3`.
