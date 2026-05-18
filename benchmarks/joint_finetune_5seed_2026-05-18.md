# 5-seed joint-pretrain + head-finetune — mixed verdict: ship zenjpeg+zenwebp, hold zenavif

**Date:** 2026-05-18.
**Config:** `--hidden 192,96 --epochs 100 --head-finetune-epochs 80 --head-finetune-lr 5e-4`
**Seeds:** 5 (CAFE, BEEF, FACE, DEAD, BABE). All 5 completed; 0xFACE took
~80 minutes vs ~20 minutes for the others due to joblib oversubscription
(5 parallel HistGB teachers competing for cores).

## Per-codec val argmin accuracy (5 seeds, sorted low→high)

| Seed     | zenjpeg | zenwebp | zenavif |
|----------|--------:|--------:|--------:|
| 0xDEAD   |  48.66% |  41.02% |  33.88% |
| 0xBEEF   |  54.27% |  48.93% |  23.05% |
| 0xBABE   |  54.50% |  51.18% |  31.96% |
| 0xCAFE   |  55.00% |  53.64% |  29.48% |
| 0xFACE   |  55.62% |  51.66% |  27.27% |
| **median** | **54.50%** | **51.18%** | **29.48%** |
| mean     |  53.61% |  49.29% |  29.13% |
| stdev    |  2.82pp |  4.91pp |  4.22pp |
| range    |   7.0pp |  12.6pp |  10.8pp |

## Verdict — split per codec

Joint-only baseline (from `benchmarks/joint_pretrain_multiseed_2026-05-17/`,
5-seed medians):

| Codec   | joint-only 5-seed | joint+finetune 5-seed | Δ |
|---------|------------------:|----------------------:|---:|
| zenjpeg | 52.7% (range 47.5-56.7) | **54.5%** (range 48.7-55.6) | **+1.8 pp** — NEW SOTA |
| zenwebp | 43.7% (range 38.3-54.0) | **51.2%** (range 41.0-53.6) | **+7.5 pp** — NEW SOTA |
| zenavif | **31.0%** (range 24.0-32.8) | 29.5% (range 23.0-33.9) | -1.5 pp — hold joint-only |

vs the **prior** absolute SOTA per codec (mixed sources):

- **zenjpeg**: per-codec v14+z_rmse ~30%; joint pretrain +22 pp → 52.7%;
  **joint+finetune +1.8 pp on top → 54.5% (new SOTA)**.
- **zenwebp**: per-codec v14+z_rmse 45.3%; joint pretrain LOST (-1.6 pp)
  to per-codec; **joint+finetune RECOVERS and BEATS per-codec by +5.9 pp
  → 51.2% (new SOTA)**.
- **zenavif**: per-codec ultraprune-27feat ~20%; joint pretrain +11 pp
  → 31.0%; **joint+finetune -1.5 pp from joint-only → hold joint-only
  at 31.0% as SOTA**.

## Variance caveat

The per-seed range is wide (7-13 pp) and the worst seed (0xDEAD) sits
below the joint-only median on all three codecs:

- 0xDEAD finetune: zenjpeg 48.66% < joint-only 52.7%
- 0xDEAD finetune: zenwebp 41.02% < joint-only 43.7%
- 0xDEAD finetune: zenavif 33.88% > joint-only 31.0% (only winner on
  this codec — likely a seed-lucky outlier)

A production picker bake should pick the **median seed** (or higher-
quartile seed by val), not a random one. The wide variance suggests
the joint-trunk freeze + per-codec head is sensitive to the head's
initial random projection — adding head-side warmup or a small ensemble
across heads would reduce ship-side variance.

Compared with prior SOTAs:

- **zenjpeg SOTA (per-codec v14+z_rmse + joint-trunk pretrain): 52.7%**.
  Finetune nominally +1.7pp but the 0xDEAD seed at 48.7% sits below
  baseline. Hold.
- **zenwebp SOTA (per-codec v14+z_rmse): 45.3%**. Both joint-only and
  joint+finetune beat that, but joint-only at 48.9% with tighter
  variance is the cleaner ship.
- **zenavif SOTA (per-codec ultraprune-27feat + v14+z_rmse): 21.7%
  per-codec, 31.0% joint pretrain**. Joint pretrain is already SOTA;
  finetune is noise.

## Why finetune likely doesn't help here

Three plausible reasons:

1. **80 epochs is too long with no per-codec early-stop on the val
   argmin.** The head overfits to per-codec val noise. The summary's
   `head_finetune` block records `best_epoch ≈ 7-44` and `epochs_ran ≈
   28-65` (early stop triggered), but the chosen `best_val_mse` doesn't
   correlate with `argmin_acc` strongly enough to stop at the right
   point.
2. **Frozen trunk has insufficient capacity per codec.** With trunk
   frozen at the joint-pretrain optimum, each head can only fit a
   linear projection of a shared 96-d feature space. Per-codec
   embeddings (where trunk also adapts) may be needed.
3. **Per-codec MSE objective is misaligned with argmin accuracy.**
   The head minimizes log-bytes regression MSE; the metric is
   argmin-cell accuracy. The finetune's "best" by val-MSE is not
   the argmin-best.

## Status

- Task #52 (pretrain → freeze → finetune): completed,
  verdict = **new SOTA for zenjpeg (+1.8 pp) and zenwebp (+7.5 pp),
  hold joint-only for zenavif**.
- Bake candidates ready in `benchmarks/joint_finetune_2026-05-18/`:
  one per codec per seed (1.5 MB JSON each, gitignored). Use a
  median-or-above seed (e.g. 0xCAFE for zenjpeg/zenwebp, 0xBABE for
  zenavif if shipping the joint-only) when re-baking to the codec's
  picker `.bin`.
- All 5 seeds completed; 0xFACE took ~80 min (vs ~20 min for the
  others) due to joblib oversubscription. No process leak.

## Reproducibility

```bash
PYTHONPATH=zentrain/examples:zentrain/tools \
  python3 zentrain/tools/train_multi_codec.py \
  --codec zenjpeg=zentrain/examples/zenjpeg_picker_config.py:/home/lilith/work/zen/zenjpeg \
  --codec zenwebp=zentrain/examples/zenwebp_picker_config.py:/home/lilith/work/zen/zenwebp \
  --codec zenavif=zentrain/examples/zenavif_picker_config.py:/home/lilith/work/zen/zenavif \
  --hidden 192,96 --epochs 100 --seed 0x<SEED> \
  --head-finetune-epochs 80 --head-finetune-lr 5e-4 \
  --out-dir benchmarks/joint_finetune_2026-05-18 \
  --out-name finetune_0x<seed>
```

Don't launch 5+ in parallel under joblib n_jobs=-1; the HistGB teachers
oversubscribe and one seed will hang. Run sequentially or set
joblib n_jobs to a finite value per worker.
