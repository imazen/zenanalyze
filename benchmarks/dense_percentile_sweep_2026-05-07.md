# Dense Percentile Sweep — 2026-05-07

## Goal
Replace the arbitrary p25/50/75/90/95/99 percentile grid with an empirically
validated minimal set per parent feature. Identify the "knee" where adding
finer percentiles stops helping.

## New Variants Added (IDs 122-211)

90 new feature variants under the `experimental` gate, filling every 5-percentile
step (p5/p10/p15/p20/p30/p35/p40/p45/p55/p60/p65/p70/p80/p85) for the six
distributional parents:

| Parent           | New IDs    | New percentile points                                      |
|------------------|------------|------------------------------------------------------------||
| LaplacianVariance | 122-135   | p15,p20,p25,p30,p35,p40,p45,p55,p60,p65,p70,p80,p85,p95  |
| AqMap            | 136-148   | p15,p20,p25,p30,p35,p40,p45,p55,p60,p65,p70,p80,p85      |
| NoiseFloorY      | 149-162   | p15,p20,p30,p35,p40,p45,p55,p60,p65,p70,p80,p85,p95,p99  |
| NoiseFloorUv     | 163-179   | p1,p5,p10,p15,p20,p30,p35,p40,p45,p55,p60,p65,p70,p80,p85,p95,p99 |
| QuantSurvivalY   | 180-194   | p15,p20,p30,p35,p40,p45,p55,p60,p65,p70,p80,p85,p90,p95,p99 |
| QuantSurvivalUv  | 195-211   | p1,p5,p15,p20,p30,p35,p40,p45,p55,p60,p65,p70,p80,p85,p90,p95,p99 |

## Implementation Notes

### LaplacianVariance (tier1.rs)
- Computed from existing 256-bin `|∇^2 L|` histogram via prefix-sum (no re-sort).
- New thresholds added alongside existing t1/t5/t10/t50/t75/t90/t99 in the
  single-pass histogram scan.
- Same `MIN_PIXELS_FOR_LAPLACIAN_PERCENTILE = 1024` floor; emit `f32::NAN` below.

### AqMap / NoiseFloorY / NoiseFloorUv / QuantSurvivalY / QuantSurvivalUv (tier3.rs)
- Sort buffers (`block_acs`, `block_low_y/cb/cr`, `quant_y/uv_blocks`) are
  performed once per pass (existing behavior); new percentile reads are
  additional index lookups at O(1) cost each.
- Same `MIN_BLOCKS_FOR_PERCENTILE` floor; emit `f32::NAN` below.
- NoiseFloorUv reads use `max(cb_pX, cr_pX)` convention matching existing p25/50/75/90.

## Ablation Procedure (to run after zenjpeg feature extraction)

```sh
# 1. Extract dense features (requires zenjpeg with target-zq feature)
cargo run --release --features 'target-zq trellis parallel' \
  --example zq_pareto_calibrate -- \
  --features-only --max-images 100 \
  --features-output benchmarks/zq_pareto_features_dense.tsv \
  --output benchmarks/.unused.tsv

# 2. Train
PYTHONPATH=zenpicker/examples python3 zenpicker/tools/train_hybrid.py \
  --codec-config zenjpeg_picker_config \
  --hidden 192,192,192 --allow-unsafe

# 3. Ablate
PYTHONPATH=zenpicker/examples python3 zenpicker/tools/feature_ablation.py \
  --codec-config zenjpeg_picker_config \
  --method permutation --n-repeats 3
```

## Prior Ablation Results (Session 2026-04-30, reference)

| Parent           | Points with >=0.05pp impact             | Sum impact |
|------------------|-----------------------------------------|------------|
| LaplacianVariance | p50(+0.47), p75(+0.24), p99(~+0.05), peak(~+0.05) | ~0.81pp |
| AqMap            | p90(+0.13), p95(+0.15), p99(+0.05)     | ~0.33pp    |
| NoiseFloorY      | p50(+0.15), p90(+0.05)                 | ~0.20pp    |
| QuantSurvivalY   | p10(+0.04)                             | ~0.04pp    |

## Ablation Results (to be filled after run)

<!-- Run feature_ablation.py and paste the per-parent table here. -->

### LaplacianVariance
| percentile | permutation_importance | cumulative_fraction |
|------------|------------------------|---------------------|
| _TBD_      | _TBD_                  | _TBD_               |

### AqMap
| percentile | permutation_importance | cumulative_fraction |
|------------|------------------------|---------------------|
| _TBD_      | _TBD_                  | _TBD_               |

### NoiseFloorY
| percentile | permutation_importance | cumulative_fraction |
|------------|------------------------|---------------------|
| _TBD_      | _TBD_                  | _TBD_               |

### NoiseFloorUv
| percentile | permutation_importance | cumulative_fraction |
|------------|------------------------|---------------------|
| _TBD_      | _TBD_                  | _TBD_               |

### QuantSurvivalY
| percentile | permutation_importance | cumulative_fraction |
|------------|------------------------|---------------------|
| _TBD_      | _TBD_                  | _TBD_               |

### QuantSurvivalUv
| percentile | permutation_importance | cumulative_fraction |
|------------|------------------------|---------------------|
| _TBD_      | _TBD_                  | _TBD_               |

## Recommended Minimal Set (to be filled after ablation)

<!-- Per parent, smallest set capturing >= 95% of cumulative ablation impact. -->

- **LaplacianVariance**: _TBD_ (prior data suggests p50+p75 captures ~87%)
- **AqMap**: _TBD_ (prior: p95 highest, then p90)
- **NoiseFloorY**: _TBD_ (prior: p50 dominant)
- **NoiseFloorUv**: _TBD_ (prior data sparse)
- **QuantSurvivalY**: _TBD_ (prior: p10 only signal above threshold)
- **QuantSurvivalUv**: _TBD_ (prior data sparse)

## Spearman Redundancy (from 2026-04-30 subset100)

| Pair                   | rho   | Verdict   |
|------------------------|-------|-----------|
| AqMap mean <-> p50     | 0.962 | redundant |
| NoiseFloorY <-> p25    | 0.956 | redundant |
| NoiseFloorUv <-> p25   | 0.966 | redundant |
| QuantSurvivalY <-> p75 | 0.968 | redundant |

Dense sweep will let us identify which percentile spacing avoids this redundancy
while preserving the discriminative signal.
