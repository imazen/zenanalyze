# imazen-26 SDR training corpus — rendered renditions + features (2026-06-14)

The budget-selected variants (`imazen26_budget_select_2026-06-14.md`, 1.5 GP
prefix) rendered to actual image files via **zenresize Mitchell-sharp**, with
zenanalyze features re-extracted on the Mitchell renditions (so the training
features describe the exact pixels that get encoded, not the Lanczos3 proxy used
during selection).

## Artifacts

| | |
|---|---|
| **Renditions** | `/mnt/v/output/imazen-26-features/train_renditions_2026-06-14/` — **1482 PNGs, 1.7 GB**, named `<even-id>.scale<W>x<H>.png` |
| **Features (canonical)** | `/mnt/v/output/imazen-26-features/imazen26_train_features_2026-06-14.parquet` — 1482 rows × 114 cols (4 meta + 110 features), 654 KB, sha256-16 `7f6f59b2760caebb` |
| Raw features TSV | `…/imazen26_train_features_2026-06-14.tsv` (1.5 MB, kept) |
| Variant manifest | `…/imazen26_train_variants_2026-06-14.tsv` (FPS order; the selection) |
| **Tower mirror** | `/mnt/tower/output/imazen-26-features/` (features parquet + variant manifest) |

The 1.7 GB renditions stay on `/mnt/v` only — they regenerate deterministically
in ~56 s from the committed manifest + tool (below), so they're not Tower-mirrored.

## How rendered

- `examples/render_imazen26_variants.rs` (zenresize 0.3.1 dev-dep), threaded.
- `Filter::Mitchell` + `resize_sharpen(10.0)` (imageflow-compatible; natural≈6.7%).
  Native renditions (target == source dims) are copied, not resampled/sharpened.
- All 110 features via `--features experimental,hdr` (`FeatureSet::SUPPORTED`).
- 1482/1482 rendered, 0 failures, 56 s, 24 threads.

## Scope

- **SDR only** — sources are the `.sdr.png` mirror. HDR renditions (UltraHDR/heic)
  are blocked on the zenpipe S-stack publish (zenpixels 0.2.14 → zencodec 0.1.23
  → heic 0.2.0); see the zenpipe `wip/heic-decode-blocked` bookmark + #48.
- Even-id images only (odd = hold-out); the id is preserved in every filename, so
  the train/hold-out split propagates with no leakage.

## Regenerate

```bash
cargo run --release --features experimental,hdr --example render_imazen26_variants -- \
  --manifest    /mnt/v/output/imazen-26-features/imazen26_train_variants_2026-06-14.tsv \
  --out-dir     /mnt/v/output/imazen-26-features/train_renditions_2026-06-14 \
  --features-out /mnt/v/output/imazen-26-features/imazen26_train_features_2026-06-14.tsv \
  --max-gp 1.5   # truncate the FPS-ordered manifest at any gigapixel budget
```
