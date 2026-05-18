# Inter-metric agreement and cross-target overhead (2026-05-18)

## Data found on R2 (user pointed me at this)

R2 bucket `s3://zentrain/` has:
- `sweep-vNN-YYYY-MM-DD/<codec>/` — raw sweep TSVs with `encoded_bytes`,
  `score_zensim`, `score_ssim2_gpu`, `score_butteraugli_*_gpu` columns
- `cvvdp-backfill-2026-05-15-half/cvvdp_imazen/` — per-sweep cvvdp scores
  keyed on `(image_path, codec, q, knob_tuple_json)`
- `ssim2-backfill-2026-05-18/ssim2_imazen/` — per-sweep ssim2 scores
- `unified-2026-05-07/` — pre-joined per-codec parquets

Local copy at `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/` has the
v12/v13/v15rc joins. Inspected.

## Inter-metric agreement (v13 zenjpeg, 36k rows / 200 images / 36 configs / 5 q)

For each image, find the bytes-optimal config whose metric score ≥ q-quantile of that
metric over that image's configs. Then check how often the answer matches across
metrics:

| quantile | zensim↔ssim2 | zensim↔cvvdp | ssim2↔cvvdp |
|---:|---:|---:|---:|
| 0.50 | 60.0% | 19.0% | 25.5% |
| 0.70 | 32.0% | 24.5% | 24.0% |
| 0.85 | 35.5% | 29.0% | 24.0% |
| 0.95 | 68.5% | 66.0% | 54.0% |

zensim and ssim2 agree ~30-60% in the medium-quality range. cvvdp diverges from both
~24-29% agreement. At the very-top-quality target (q=0.95) all three converge because
only the highest-q configs survive each metric's filter.

## Cross-target bytes overhead (v13 zenjpeg, q=0.85)

If picker optimizes for X but the user evaluates with Y, what is the bytes penalty?

| opt→eval | median % | p90 % | p95 % | pick reaches Y |
|---|---:|---:|---:|---:|
| zensim→ssim2 | 0.00% | 1.35% | **1.51%** | 60.3% |
| zensim→cvvdp | 0.00% | 31.59% | **38.46%** | 74.5% |
| ssim2→zensim | 0.00% | 6.13% | **11.28%** | 73.4% |
| ssim2→cvvdp | 0.01% | 32.43% | **36.79%** | 75.9% |
| cvvdp→zensim | 0.00% | 6.17% | **6.45%** | 50.0% |
| cvvdp→ssim2 | 0.00% | 2.85% | **6.29%** | 41.7% |

**Key reading**: median overhead is 0% across every pair (at least half the time the
bytes-optimal config is the same regardless of metric). The tails matter:

- **zensim-trained picker → cvvdp users**: p95 = 38.46% bytes overhead. The worst
  5% of images pay 38% extra bytes if the user actually cares about cvvdp.
- **ssim2-trained picker → cvvdp users**: same problem (p95 = 36.79%).
- **cvvdp-trained picker → zensim or ssim2 users**: p95 = 6.45% / 6.29%. The reverse
  direction is much friendlier.
- **cvvdp picker hit-rate is lower** (41-50% reaches Y threshold) — cvvdp tends to
  rate configs lower, so a picker optimizing for it tends to recommend higher-bytes
  configs that may still miss the other metrics' top-quantile thresholds.

## What this implies

1. **If the production user cares about cvvdp** (closest to human MOS per
   ColorVideoVDP's calibration), today's zensim-trained picker is suboptimal on
   the worst 5-10% of images at the ~35-40% bytes scale. That's not a noise-level
   gap — it's enough to matter.

2. **If the production user cares about ssim2**, the zensim-trained picker is
   already close-enough on the bulk of images (p95 ~1.5%). zensim's PreviewV0_2 is
   ssim2-derived, so this is consistent.

3. **A cvvdp-targeted picker is more "conservative"** (over-allocates bytes,
   misses tight thresholds more often) — would need careful threshold calibration
   if shipped.

## What would actually be needed to retarget

Real-shipping-grid picker bakes against ssim2 / cvvdp need:
1. **For zenjpeg**: v15rc parquet (513K rows, 901 images, 30 knobs) has zensim + ssim2
   but cvvdp is all-NaN there. Either backfill cvvdp on v15rc, or use the smaller v13
   data (36k rows, gen-chart only) which has all 3.
2. **For zenwebp**: v12 unified parquet (1k rows) has all 3 but is tiny. v05c+ has
   wider sweeps but only zensim. Need wider × multi-metric.
3. **For zenavif**: v05c has 47MB / ~600k rows on R2 but ssim2/dssim columns are
   sparse (backfill incomplete).

**Concrete unblock**: backfill cvvdp + ssim2 on the v15r/v15rc zenjpeg sweep and on
the latest zenwebp/zenavif sweeps. Then a 1-hour multi-seed-confirm per codec gives
ssim2-target vs cvvdp-target vs zensim-target pickers.

## Recommendation

Track this in a follow-up task. For this autonomous block, continue on #52-#55 with
zensim as the operating target. The inter-metric finding is the science answer to
"does the picker generalize across metrics?" — **no, the median is fine but the
tails diverge sharply, particularly for cvvdp**.

## Side-task #54: bake compression on joint-pretrained per-codec models

The new `bake_picker.py --zerobias-tau 0.005 --compress --optimize` flags were
applied to the joint pretrain outputs (seed 0xcafe). Sizes per codec:

| Codec | f32 bare | f32 compressed | f16+optimize | i8+optimize |
|---|--:|--:|--:|--:|
| zenjpeg | 200,424 | 193,034 | 99,362 | **53,032** |
| zenwebp | 195,264 | (similar shrink) | 97,473 | **51,889** |
| zenavif | 189,092 | (similar shrink) | 95,567 | **50,677** |

i8+optimize gives ~52 KB per codec bake (3 codecs × 52 KB ≈ 156 KB total joint pretrain
shipping payload). zenpredict 0.2.x's lz4 transparent decode supports this directly.

**Note**: zerobias at the calibrated `τ=0.005` only zeros 6% of weights in the joint
pretrained MLPs because the joint trunk has more concentrated weight distribution
(|max| ≈ 1.05) than V_18's zensim bake (|max| ≈ 10-100). For the joint pretrain bakes,
zerobias contributes less than it did for V_18; the i8 quantization itself + LZ4 is
where the compression comes from. `τ=0.02` would zero ~24% but its SROCC delta on this
methodology isn't measured.
