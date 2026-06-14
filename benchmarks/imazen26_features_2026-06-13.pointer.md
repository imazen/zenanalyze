# imazen-26 full-feature extraction (2026-06-13) — data pointer

All 110 `main`-branch features (`experimental` + `hdr`) extracted over the
imazen-26 corpus **and zoom/region crop + resize derivatives**. Large data
lives in block storage + Tower, NOT git (per the ML-data discipline).

## Artifact

| | |
|---|---|
| **Parquet (canonical)** | `/mnt/v/output/imazen-26-features/imazen26_features_2026-06-13.parquet` |
| sha256 | `aab7c7d2f06687a069283f0e43b9fc31087f216a3d6120b12231d0ef619af774` |
| size / shape | 61 MB · **246,819 rows × 119 cols** (9 meta + 110 `feat_*`) |
| Raw TSV | `…/imazen26_features_2026-06-13.tsv` (280 MB, 4.6× larger — kept alongside) |
| Manifest | `/mnt/v/output/imazen-26-features/imazen26_manifest.tsv` (sha256 `fa36092485eed4afdc09ed340b8f46c8da81200d6659054b3f9a5d34a15ba10c`) |
| **Tower mirror** | `/mnt/tower/output/imazen-26-features/` (parquet + manifest; parquet sha256-verified ✓) |

## Provenance

- **Corpus:** imazen-26 PNG-normalized (`/mnt/v/output/imazen-26-png`, SDR
  `.sdr.png`), **2157 / 2160** manifest images (3 HDR-only night shots failed
  SDR conversion), 21 content classes. Source manifest:
  `~/work/codec-corpus/imazen-26/CORPUS-MANIFEST.tsv`.
- **Extractor:** `examples/extract_features_imazen26_crops.rs`, built
  `--features experimental,hdr` → `FeatureSet::SUPPORTED` (all 110 features
  incl. `palette_density`, `xyb444_color_loss`, `xyb_bquarter_chroma_loss`,
  HDR/depth). `std::thread::scope` parallel; output verified byte-identical to
  single-threaded. 0 decode failures.
- **Crop derivatives (`crop_label`):** `full` + `c50_{center,tl,tr,bl,br}` +
  `c25_{center,tl,tr,bl,br}` — a 50%×50% and 25%×25% aspect-preserving window
  at center + 4 corners (11 variants/image).
- **Resize grid (`size_class`):** dense log-spaced maxdim
  `{32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096}` + `native`,
  Lanczos3, **downscale-only** (a target ≥ a variant's maxdim is skipped — the
  `native` row covers full size; never an upscale).
- **Schema:** `image_path, image_sha (=corpus number), split (85/15 train/val
  by hash), content_class, source, crop_label, size_class, width, height,
  feat_<name>…`. NaN percentile sentinels (too few samples) → empty cell.

## Scope note — 110 vs 211 features

110 = every feature on `main`. The extra **90 dense-percentile features
(ids 122–211) exist only on the unmerged, quarantined `feat/dense-percentiles`
branch** (see `everything.md`). To include them, re-run this extractor on that
branch (a second pass); this artifact covers all of `main`.

## Regenerate

```bash
cargo run --release --features experimental,hdr \
  --example extract_features_imazen26_crops -- \
  --manifest /mnt/v/output/imazen-26-features/imazen26_manifest.tsv \
  --output   /mnt/v/output/imazen-26-features/imazen26_features_<DATE>.tsv
python3 benchmarks/tsv_to_parquet.py <…>.tsv --keep-tsv
```
The manifest is rebuilt from `CORPUS-MANIFEST.tsv` by mapping each `path` to
`/mnt/v/output/imazen-26-png/<folder>/<stem>.sdr.png`.
