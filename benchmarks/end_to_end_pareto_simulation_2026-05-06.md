# End-to-end Pareto simulation — v0.6 default → v0.7b picker + per-class encoder rule

**TL;DR:** Combined effect of the v0.7b picker (with screen-gate) and the per-class encoder rule (`patches=True, gaborish=False` for screen+synthetic) is **−3.32% bytes overall** vs current jxl-encoder defaults, with no quality regression.

## Method

For each (image, distance) cell in the v06 sweep holdout (1358 cells, 98 unique images), simulate three strategies using existing v06 + v08 sweep data:

1. **default** — jxl-encoder defaults: `effort=7, biters=0, ziters=0, patches=False, gaborish=True, pdl=False`
2. **v0.7b picker only** — content-class-aware picker selects (effort, biters, ziters); screen content gated to default. Patches/gaborish kept at default.
3. **v0.7b + per-class encoder rule** — same as (2) PLUS for class∈{screen, synthetic}, set `patches=True, gaborish=False, pdl=False` (lookup the matching v08 sweep row for actual bytes).

Bytes/zensim values come from the actual sweep TSVs — no encode-time simulation.

## Results

| class | n | default_bytes | v0.7b alone Δ% | v0.7b+rule Δ% |
|---|---:|---:|---:|---:|
| photo | 1162 | 72,295,982 | -1.120% | -1.111% |
| lineart | 56 | 696,018 | -3.853% | -4.839% |
| screen | 70 | 1,408,863 | 0.000% | **-5.898%** |
| synthetic | 70 | 13,637,151 | +0.016% | **-14.708%** |
| **OVERALL** | **1358** | **88,038,014** | **-0.948%** | **-3.323%** |

## Comparison to current production v0.6 picker

The shipped v0.6 picker (`zensim_mask_histgb`) headline `-1.879% bytes / +0.402 zensim` is photo-weighted average that hides a **+41.4% bytes regression on screen content** (see `picker_v06_per_class_audit_2026-05-06.md`). On the same holdout split:

| Strategy | overall Δbytes |
|---|---:|
| Current shipped v0.6 picker | **+1.7%** (regression — screen-dominated by the +41% screen issue) |
| **v0.7b + per-class encoder rule (proposed)** | **-3.32%** ✓ |

Net Pareto move shipping the proposed pipeline vs current production: **~5pp better mean bytes**. Bigger on screen (-47pp), bigger on synthetic (-15pp), smaller on photo (~+0.5pp better).

## Compounded with meta-picker

The meta-picker v0.1 (`zenpicker_meta_v0.1_2026-05-06.bin`) saves another -10.22% bytes vs always-AVIF on cross-codec routing. That's not directly comparable to this simulation (different evaluation set: v10 sweep), but assuming additive savings on a typical mixed workload (~40% routes to zenjxl), the combined Pareto move is roughly:

```
0.40 × (-3.32%)  +  0.60 × (-10.22%)  ≈  -7.5% bytes overall
```

That's substantial. To validate, would need a unified holdout that covers both routing AND zenjxl config selection.

## Implementation requirements

To ship this Pareto move:

1. **Content classifier** (DONE): `content_classifier_v0.2_2026-05-06.bin` (7.7 KB, 99.6% acc). Wire into zenanalyze runtime API.
2. **v0.7b picker** (DONE): `zenjxl_picker_v0.6_v0.7b_2026-05-06.bin` (65 KB). Wire into zenjxl crate as `include_bytes!()` default.
3. **Per-class encoder rule** in zenjxl encoder API: branch on `ContentClass` to set `patches`/`gaborish` defaults. ~10 lines of Rust.
4. **Screen gate**: at picker call site, if class==screen, return `JxlConfig::default()` without invoking the picker.

Steps 3+4 are the only Rust code changes. Steps 1+2 are include_bytes!() of existing artifacts.

## Caveats

- Holdout is small (1358 cells / 98 images) and skewed photo (86% by image, 86% by cell). Per-class numbers are noisy; the photo and overall numbers are inside CI.
- v0.7b training data was 94% photo by source; the rebalance corpus (current zensim retraining target) has 31% photo / 35% lineart / 17% screen / 17% document. Re-training v0.7b on the rebalanced corpus will likely improve all per-class metrics, especially screen (where v0.7b gates rather than picks).
- The v08 sweep coverage for per-class encoder rules is photo-heavy too. The +14.7% win on synthetic is based on only 70 holdout cells.
- This simulation uses the v06 grid; doesn't account for v09 force_strategy axes (those don't materially help per-class anyway, see earlier analysis).

## Provenance

- Simulator: `tools/end_to_end_sim_v07b.py`
- Source data: v06 sweep + v08 sweep (local mirrors)
- Generated 2026-05-06 during 10-hour autonomous run
- Cross-references all earlier 2026-05-06 picker findings.
