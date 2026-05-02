# Tier 3 LOO retrain — zenwebp paired with/without metrics, 2026-05-02

Hand-rolled driver (`benchmarks/loo_driver_2026-05-02.py`) — 10 candidate
features, 20 paired retrains, foreground execution. Each retrain trains
the same student MLP (LeakyReLU 80 → 128 → 128 → 30) on the zenwebp
v0.2 pareto sweep (~22 M cells, 1 264 unique image-instances) with
identical hyperparameters; the only change between paired runs is the
presence of a single feature in `KEEP_FEATURES`.

The base config is the live zenwebp post-cull set (36 features). For
each candidate:

- **with** = base ∪ {candidate}
- **without** = base \ {candidate}

For features already in the base (only `feat_noise_floor_y_p50`),
"with" = base unchanged, "without" = base minus it. For all others,
"without" = base unchanged, "with" = base + the candidate.

**Interpretation key:**

- ΔOH = `without_overhead − with_overhead`. Lower overhead is better,
  so **positive ΔOH** means removing the feature hurts → **keep**.
- ΔAC = `without_argmin − with_argmin`. Higher accuracy is better, so
  **negative ΔAC** means removing the feature hurts → **keep**.

## Results

| Feature | with OH | wo OH | **ΔOH (pp)** | with AC | wo AC | **ΔAC (pp)** | Verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| `feat_log_pixels` | 2.44% | 2.83% | **+0.39** | 42.5% | 43.4% | +0.90 | mixed (OH→keep, AC→cull) |
| `feat_log_padded_pixels_8` | 2.40% | 2.83% | **+0.43** | 45.2% | 43.4% | **−1.80** | **KEEP** |
| `feat_log_padded_pixels_16` | 2.16% | 2.83% | **+0.67** | 46.7% | 43.4% | **−3.30** | **KEEP** (strongest) |
| `feat_log_padded_pixels_32` | 2.14% | 2.83% | **+0.69** | 46.4% | 43.4% | **−3.00** | **KEEP** |
| `feat_log_padded_pixels_64` | 2.75% | 2.83% | +0.08 | 29.8% | 43.4% | **+13.60** | **CULL** (hurts argmin badly) |
| `feat_bitmap_bytes` | 1.93% | 2.83% | **+0.90** | 52.0% | 43.4% | **−8.60** | **RESTORE** (strongest signal in entire experiment) |
| `feat_palette_density` | 2.72% | 2.83% | +0.11 | 44.1% | 43.4% | −0.70 | weak keep |
| `feat_aq_map_p10` | 2.36% | 2.83% | **+0.47** | 42.8% | 43.4% | +0.60 | mixed |
| `feat_noise_floor_y_p10` | 2.19% | 2.83% | **+0.64** | 49.2% | 43.4% | **−5.80** | **KEEP** |
| `feat_noise_floor_y_p50` | 2.83% | 2.24% | **−0.59** | 43.4% | 43.9% | +0.50 | **CULL** |

## Headline findings

### 1. `feat_bitmap_bytes` cull was wrong — restore it

The current `BitmapBytes` cull (commit `e9cd04d0`) was motivated by
"linear in `pixel_count` for fixed-channel rgb8 corpora — the picker
MLP can scale by 3 trivially via ReLU". The data says **the strongest
single signal in the entire LOO experiment is bitmap_bytes**: ΔOH
+0.90 pp (mean overhead drops from 2.83% to 1.93% when adding it) AND
ΔAC −8.6 pp (argmin accuracy jumps from 43.4% to 52.0%). Even though
the channel/byte multipliers are constant on rgb8, the model uses
this column as a useful numerical handle that scales differently from
`pixel_count`. The cull is empirically incorrect for tiny MLPs — same
class of error as the `LogPixels` cull we already reverted.

### 2. `feat_log_padded_pixels_64` actively hurts

Adding it drops argmin accuracy by 13.6 pp. This is **larger than any
positive contribution** in the experiment. The likely cause: at our
data scale (1 264 unique image-instances spread across 6 cells, ~210
per cell), the 64×64 grid signal is mostly aliased to `LogPixels`
except on the very largest images, but the model spends parameters on
it anyway. The information ROI is negative.

The `log_padded_pixels_{8,16,32}` triplet all consistently help
(ΔAC −1.8 to −3.3 pp), so the value comes from the **smaller transform
grids that match WebP's 4×4 / 8×8 hybrid + JXL DCT16/32**. The 64×64
grid is past the useful resolution for our codec set.

### 3. `feat_noise_floor_y_p50` should be culled

The only feature already in the base — and the only one whose removal
genuinely *helps* (ΔOH −0.59 pp, ΔAC +0.5 pp). Confirms the 4-codec
Tier 0 finding that it clusters tightly with `feat_aq_map_p50`. Safe
to cull.

### 4. `feat_log_padded_pixels_{8,16,32}` and `feat_noise_floor_y_p10` clearly earn their keep

ΔOH +0.43 to +0.69 pp; ΔAC −1.8 to −5.8 pp. Strong evidence the
tiny-MLP expressivity argument was right for these — the model uses
the block-grid log scaling directly.

### 5. Mixed signals: `feat_log_pixels`, `feat_aq_map_p10`, `feat_palette_density`

OH drops slightly when adding them (helpful) but argmin accuracy
either flat or marginally worse. Likely each is independently weak but
not actively harmful. Single-seed N=1 — variance across seeds could
easily be ±2 pp on argmin. Need multi-seed runs before any cull.

### 6. `feat_log_pixels` restoration: the data is weaker than I claimed

The earlier restore commit (`15d9c299`) was on theoretical grounds
("tiny MLPs can't recover log"). The empirical signal is +0.39 pp OH
(helpful) but +0.90 pp argmin (hurtful). The restoration may have
been a wash — not actively wrong, but also not the win the
permutation-importance finding suggested. Multi-seed retrain would
clarify.

## Recommended next moves

**High-confidence (single-seed, ΔAC ≥ 1.5 pp magnitude on at least
one metric):**

1. **Restore `feat_bitmap_bytes`** — strongest signal in experiment.
2. **Cull `feat_log_padded_pixels_64`** — actively hurts at our scale.
3. **Cull `feat_noise_floor_y_p50`** — marginal improvement when
   removed, consistent with Tier 0 cluster anchor finding.

**Keep — confirmed by data:**

4. `feat_log_padded_pixels_{8,16,32}` (currently restored).
5. `feat_noise_floor_y_p10` (currently kept).

**Defer pending multi-seed:**

6. `feat_log_pixels` — restore stays for now; mixed single-seed signal.
7. `feat_aq_map_p10` — mixed.
8. `feat_palette_density` — weak positive; keep until disproven.

## Caveats

- **Single-seed N=1.** No variance estimate. Should re-run with seeds
  {0, 1, 2, 3, 4} before any production cull, especially for the
  weak-signal candidates.
- **Single codec (zenwebp).** zenjpeg's pareto TSV doesn't have
  `log_padded_pixels_*`; zenjxl/zenavif similar. Cross-codec
  confirmation needed.
- **Single base config.** The exact ΔAC could shift if the base
  changes (some features may be redundant with each other in
  combinations not tested here).
- **train_hybrid.py uses a fixed 80/20 train/val split keyed by
  image_path.** Multiple runs with the same data give the same
  split — variance comes only from the MLP init seed, not data
  resampling. To get cross-validated estimates, the trainer would
  need a `--cv-folds N` mode it doesn't currently have.
- **Argmin accuracy is the more sensitive metric** in this
  experiment; it caught both the bitmap_bytes restoration signal
  and the log_padded_pixels_64 cost. Mean overhead is more stable
  but discriminates less.

## Artifacts

- `benchmarks/loo_driver_2026-05-02.py` — driver source
- `benchmarks/loo_retrain_2026-05-02.tsv` — paired metrics, machine-readable
- `/tmp/loo_handrolled_2026-05-02/` — per-feature train logs (10 features
  × 2 retrains = 20 logs); not committed (large, reproducible from
  driver)

Total wall: ~17 min for 20 retrains on the 7950X.
