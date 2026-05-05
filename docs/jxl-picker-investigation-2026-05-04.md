# zenjxl picker investigation — 2026-05-04

**Question**: are we using the wrong metric? are we fighting that? do we need
to expand or shrink codec parameters? — should at least break even, ideally
parity-while-faster.

**Result**: confirmed there's room to win. Two independent axes of fix
identified, ranked by cost.

## TL;DR

1. **The v0.5 picker's HOLD verdict was teacher-labeling bug, not data
   shortage.** A simple HistGradientBoostingClassifier on the same v05c
   data, retrained with safety-constrained teacher labels, gets
   **-1.78% bytes / +0.22pp zensim / -2.5% encode time** on a held-out
   248-image split. v0.5 was at +3.11% bytes / +1.62pp zensim — the
   correction is +5pp on bytes for free.

2. **Expanded JXL knobs (butteraugli_iters etc.) help in the
   speed-unconstrained regime (~67% wins, mean -4% bytes), but not in
   the speed-safe regime** that the user explicitly named. For
   "parity while encoding faster" the v05c data alone is sufficient.

3. **zen-metrics CLI now exposes the expert knobs** (PR #2 on
   imazen/turbo-metrics) for future v06 sweeps targeting the
   speed-unconstrained quality regime, but no v06 sweep is required to
   ship a picker improvement.

## Investigation chain

### v05c oracle: only 9.1% of cells have a speed-safe alternative

Of 31,422 (image, distance) cells in v05c (noise=False only):
- **2,857 cells (9.1%)** have a different effort that beats effort=7 on
  bytes (≥1% smaller) without losing zensim (≥-0.05pp) or encode time
  (≤+5%).
- For those 9.1%: mean -7.91% bytes / +1.22pp zensim / -67% encode time.
- The remaining 90.9% of cells: effort=7 IS the optimal choice.

**The v0.5 picker (60% effort=3) was wrong on the 91% majority.**

Top "speed-safe wins" cell distribution: effort=5 (1420x), effort=3
(1336x), effort=9 (101x). Effort=5 wins are concentrated at distance ≥
1.0; effort=3 wins are concentrated at very tight distances.

### Distance-aware classifier picker prototype

Built a tiny prototype that:
- Reads v05c features + sweep TSV
- Per (image, distance), labels the picker target as: argmin(bytes) over
  cells where (zensim ≥ default - 0.05) AND (encode_ms ≤ default × 1.05).
  Default = effort=7. If no cell beats default, label = effort=7.
- HistGradientBoostingClassifier with class_weight=balanced on
  (features ⊕ log(distance)).
- Image-level 80/20 holdout (no leakage), seed=7.

Results on 4712 held-out cells across 248 images:

| Metric | Value |
|---|---:|
| Mean Δbytes vs default | **-1.78%** |
| Median Δbytes vs default | 0.000% (>50% picks default) |
| Mean Δzensim | +0.22pp |
| Mean Δencode_ms | -2.53% |
| Pick rate: default (effort=7) | 53.2% |
| Pick rate: safe-alt | 46.8% |
| Oracle had safe-alt available | 29.3% |
| Picker matched oracle | 67.2% |

Per-distance band:
- tight (≤1.0): -2.02% bytes / +0.14pp zensim / -9.4% ms
- mid (1..3): -1.62% bytes / +0.17pp zensim / +17% ms
- loose (>3): -1.65% bytes / +0.33pp zensim / -9.7% ms

Picker over-picks safe-alt (47% vs oracle's 29%) — somewhat aggressive
but the conservative pessimistic structure (default=effort 7) keeps the
A/B safe.

vs v0.5 picker (HOLD verdict): +3.11% bytes / +1.62pp zensim. The
prototype is **+5pp better on bytes** by changing only the teacher
labeling.

### zen-metrics CLI knob expansion (turbo-metrics PR #2)

The JXL sweep CLI now exposes the LossyConfig knobs the zenjxl wrapper
hides:

- `butteraugli_iters` / `zensim_iters` / `ssim2_iters` — iterative
  metric-targeted refinement (gated by jxl-encoder cargo features
  `butteraugli-loop`, `ssim2-loop`, `zensim-loop`).
- `pixel_domain_loss`, `patches`, `gaborish`, `error_diffusion`,
  `denoise`, `lf_frame`, `lz77`, `progressive`.
- `force_strategy`, `max_strategy_size` (DCT strategy).

Why this matters: in jxl-encoder, `effort=N` is a macro-knob that
bundles butteraugli_iters, adaptive_quant, DCT16/32/64, LZ77 method,
fine_grained_step, initial_q. Sweeping `effort × distance` (v05c) means
the picker can never select e.g. "effort=5's DCT search + 2 butteraugli
iters + no LZ77" cells.

Single-image local validation found `(effort=5, butteraugli_iters=2)`
beats `(effort=9, default biters=4)` on bytes (-0.94%), zensim
(+0.08pp), AND speed (1.65× faster).

25-image stratified validation sweep at distance ∈ {0.5, 1.0, 3.0} ×
effort ∈ {3,5,7,9} × biters ∈ {0,1,2,4}:
- Speed-safe (≤5% slower) wins: 32/75 = 43%, mean -3.32% bytes,
  +0.40pp zensim, -61% encode time.
- Speed-unconstrained: 50/75 = 67%, mean -3.97% bytes, +0.29pp zensim,
  +103% encode time.

**Top speed-safe wins are 100% biters=0** — the iterative refinement
knobs don't help in the parity-with-faster regime. They DO help when
willing to pay 2-4× encode time for an additional ~3% bytes.

53/1248 cells failed decode — `effort=9 + biters=4` triggers a
`Invalid AC: 1 nonzeros after decoding block` jxl-encoder bug on
gb82-screen content. Worth filing; not blocking.

## Path forward (recommended order)

1. **Free fix (no compute)**: Productionize the distance-aware picker
   on v05c data. Wrap as a proper picker artifact (ZNPR v3 binary)
   that the zenjxl runtime can load.
2. **Optional polish ($0)**: Try other classifier choices (XGBoost,
   tiny MLP) to push picker accuracy from 67% → 75%+.
3. **Cheap compute ($5-10)**: Run a v06 sweep with butteraugli_iters
   added on a clustered 300-image subset. Train a second picker for
   the speed-unconstrained "highest quality" tier.
4. **Skip**: A full v06 sweep with the entire expert knob surface
   (~228 cells/image × 700 images = 160k cells). Most expert knobs
   (pixel_domain_loss, patches, gaborish) give ≤1% deltas on typical
   content; not worth the compute until the speed-safe path is
   shipped first.

## Artifacts

- v05c oracle script: `/home/lilith/sweep-data/oracle_v05c_zenjxl.py`
- Distance-aware picker prototype:
  `/home/lilith/sweep-data/picker_v06_proto.py`
- Local validation sweep TSV: `/tmp/jxl_v06_local/sweep_biters.tsv`
- v06 grid generator (turbo-metrics):
  `scripts/sweep/generate_jobspecs_v06.py`
- Expanded knob CLI: turbo-metrics commit `f43811b` on PR #2.

## Why this isn't the wrong metric

The user asked "are we using the wrong metric?" Answer: no, zensim was
the right metric for the v0.5 picker training. The bug was the teacher
signal, not the choice of metric. A separate investigation (training
against butteraugli) requires a new sweep — v05c lacks butteraugli
columns. Worth doing IF the speed-safe picker doesn't fully meet the
quality bar, but not the first thing to try.
