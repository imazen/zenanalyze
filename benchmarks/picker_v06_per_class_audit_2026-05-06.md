# v0.6 zenjxl picker — per-content-class hidden regression audit (2026-05-06)

**TL;DR:** the shipped headline `zensim_mask_histgb: -1.879% bytes / +0.402 zensim` is a **photo-weighted average** that hides a serious **+41.4% bytes regression on screen content** in holdout. The Pareto move is to gate the picker on content class — apply for photo, fall through to default for non-photo.

## Setup

Re-trained `zensim_mask_histgb` on 91% v06 sweep TSV (165k cells, 1241 images), held out 20% of images per content class. Same safety mask: `bytes < default × 0.99 AND zensim ≥ default - 0.05 AND ms ≤ default × 1.05`. Default cell: effort=7, biters=0, ziters=0.

Content class derived from filename (same heuristic as `build_cclass_tsv.py`):
- screen: terminal_/windows_/macos_/screen/gui_/browser/...
- lineart: chart/graph/diagram/logo/infographic/stock
- document: scan/document/invoice/page-
- photo: everything else

Holdout class distribution: 1246 photo / 70 screen / 56 lineart cells (98 unique images).

## Per-class results on holdout

| class | n | Δbytes% | Δzensim_pp | Δms% | default_pick% |
|---|---:|---:|---:|---:|---:|
| **lineart** | 56 | +0.000% | +0.0000 | +0.000 | 100.0% |
| **photo** | 1246 | **-0.450%** | -0.0294 | +3.796 | 85.5 |
| **screen** | 70 | **+41.445%** | -0.2658 | -32.101 | 62.9 |

**Weighted-avg Δbytes: +1.706%** (photo-dominated).

## Interpretation

- **lineart**: picker correctly defaults 100% of the time. No harm, no gain. Conservative behavior is the right move when the safety mask never activates.
- **photo**: picker wins by 0.45% bytes / -0.03 zensim — the actual headline number, just smaller than the published -1.879% because the published number was on the full training set (not holdout).
- **screen**: picker is **catastrophically bad**: it picks effort=3 instead of default effort=7 for 37% of cells, saving encode time (−32%) at the cost of bloating the file (+41%). The safety mask was supposed to prevent this, but the per-image features (DCT compressibility / variance) on screen content fool the predictor into thinking the byte savings will hold — they don't.

## Why this happens

The v0.6 picker training corpus is **94% photo by image count** (1169 photo / 38 screen / 34 lineart of 1241 unique features). The MLP / HistGB sees screens as a tiny minority and learns photo-favored decision boundaries. The safety mask filters the LABELS (which cell is the "winner") but the FEATURES still pull screens toward photo-shaped decisions.

## Pareto move

**Ship a content-class gate**: at inference time, classify the image (cheap; `zenanalyze` already provides the features). If `class != photo`, skip the picker and emit default-config. This makes:

- photo: −0.45% bytes (current win, kept)
- lineart: 0% (was 0%, no regression)
- screen: 0% (was +41%, recovered)
- weighted avg: -0.41% bytes (the photo-only result, scaled by photo fraction)

**No new training needed.** Just an inference-time gate.

## Better long-term move

Train a screen-specific picker on the rebalanced corpus (3,000+ gen-screen sources via `~/work/zen/zensim--v06-rebalance`). The data exists; the architecture is the same; the result is a class-conditioned router that can actually move the screen Pareto down (e.g. by picking patches=True for screens, which v07 explore showed dominates 42 of 117 v07-beats-v06 cells).

## Provenance

- Audit script: `tools/v06_champ_per_class.py` (kept locally; generated via this analysis)
- Source data: `~/sweep-data/zenjxl_v06.tsv` (165k cells)
- Generated 2026-05-06 during 10-hour autonomous run
- Original champion report: `picker_v06_multi_99chunks_2026-05-05.md` (without this per-class breakdown)
