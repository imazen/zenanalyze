# imazen-26 train-rendition feature re-extraction (2026-06-22) — data pointer

Re-extraction of the imazen-26 SDR **train-rendition** features on current `main`.
Supersedes `imazen26_train_corpus_2026-06-14.pointer.md` (features only — the
renders are unchanged). Block storage + Tower, NOT git.

## Artifact

| | |
|---|---|
| **Parquet (canonical)** | `/mnt/v/output/imazen-26-features/imazen26_train_features_2026-06-22.parquet` |
| sha256 | `acbb256e6fdbffbe1ab31de3c9263717af144667f08c34b7b13360698207a85e` |
| shape | 1482 × 114 (meta `variant_name, content_class, width, height` + 110 `feat_*`) |
| Raw TSV | `…/imazen26_train_features_2026-06-22.tsv` (kept alongside) |
| **Renditions** | `…/train_renditions_2026-06-14/` (1482 PNGs — **shared with the 06-14 baseline, byte-identical**, see below) |
| Variants manifest | `…/imazen26_train_variants_2026-06-14.tsv` (unchanged) |
| **Tower mirror** | `/mnt/tower/output/imazen-26-features/imazen26_train_features_2026-06-22.parquet` (sha-verified ✓) |

## Provenance

- **build_commit:** `6cb86df9`. Renderer `examples/render_imazen26_variants.rs`,
  `--features experimental,hdr`, `--max-gp 1.5 --sharpen 10.0` (zenresize Mitchell
  + `resize_sharpen(10)`). **1482 rendered, 0 failures**, 59 s.
- **Renders are byte-identical to the 2026-06-14 baseline.** `zenresize` is
  unchanged (0.3.1 → 0.3.1), so the re-render reproduced bit-exact PNGs — verified
  by a full content-hash of both dirs (`c5bb82f7…` == `c5bb82f7…`, all 1482 files).
  The freshly-rendered dir was therefore removed as a pure duplicate (1.7 GB
  reclaimed); this artifact references the canonical `train_renditions_2026-06-14/`.

## What changed vs 2026-06-14 (validated)

Renders identical → the feature changes are **purely this cycle's feature-math**
(the same 7 as the SDR main corpus), no render-driven drift. 5 are visible on this
1482-row corpus; `chroma_complexity` / `colourfulness` moved below f32 display
precision here (they showed on the larger main corpus).

| feature | rows changed | max rel-dev | cause |
|---|--:|--:|---|
| `edge_slope_stdev` | 99.5 % | 4.00 % | `rsqrt_approx` → `rsqrt_stable` |
| `variance` | 50.9 % | 0.08 % | `fixed_reduce8` |
| `chroma_luma_covariance_cr` | 7.7 % | 0.04 % | `fixed_reduce8` |
| `chroma_luma_covariance_cb` | 7.8 % | <0.01 % | `fixed_reduce8` |
| `spectral_slope_y` | 0.5 % | <0.01 % | product-then-ln |
| `chroma_complexity` / `colourfulness` | 0 (sub-precision) | — | `fixed_reduce8` (below display precision on this corpus) |

The other 103 features are byte-identical. **Downstream:** picker MLPs trained on
the 06-14 train features for the changed columns (esp. `edge_slope_stdev`) are
stale — retrain against this artifact.

## Regenerate

```bash
cargo build --release --features experimental,hdr --example render_imazen26_variants
target/release/examples/render_imazen26_variants \
  --manifest …/imazen26_train_variants_2026-06-14.tsv \
  --out-dir  …/train_renditions_<DATE> \
  --features-out …/imazen26_train_features_<DATE>.tsv \
  --max-gp 1.5 --sharpen 10.0
python3 benchmarks/tsv_to_parquet.py --keep-tsv …/imazen26_train_features_<DATE>.tsv
python3 benchmarks/validate_reextract.py <old>.parquet <new>.parquet --expect <changed feats>
```
