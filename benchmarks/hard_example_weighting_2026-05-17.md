# Hard-example weighting bench — 2026-05-17

**Verdict: REGRESSION. Default stays `none`.**

Three-seed A/B on the zenjpeg picker (`zenjpeg_picker_config`) measured
whether per-row student-vs-teacher disagreement weighting in the
LeakyReLU distill loop improves held-out argmin accuracy and mean
overhead. It does not — `emae` weighting at α=1.0, window=5, clip=10
regressed both metrics and increased seed-to-seed variance.

## Setup

- Trainer: `zentrain/tools/train_hybrid.py` @ 92a706c.
- Codec config: `zentrain/examples/zenjpeg_picker_config.py`.
- Pareto: `zq_pareto_2026-04-29.tsv` (29,185 decision rows after build).
- Features: `zq_pareto_features_2026-05-01_parallel.tsv`, 51-col schema.
- Image-level holdout: 20% (5,799 val rows / 23,386 train rows).
- Backend: `--activation leakyrelu` (PyTorch, the new default).
- Heads: bytes_log + chroma_scale + lambda × 12 cells = 48 outputs.
- 3 seeds: 1, 2, 3 (passed via `--seed`). Same Pareto, same features,
  same internal val-split logic in `_train_torch_leakyrelu_student`.

## `emae` weighting params

- `--hard-example-weighting emae`
- `--hard-example-alpha 1.0` (at the median disagreement: weight = 2.0)
- `--hard-example-ema-window 5` (α_ema = 0.2; first 5 epochs uniform)
- `--hard-example-clip 10.0` (weights ∈ [0.1, 10.0])

The implementation in `_train_torch_leakyrelu_student`:

1. Every epoch, after the parameter update, run a full-train forward
   pass and compute per-row squared MSE.
2. Blend into `disagree_ema[i]` at `α_ema = 1/window`. First
   observation initialises (no warmup mixing into NaN).
3. After `hard_example_ema_window` epochs (≥ 5), compute
   `row_weights[i] = clip(1 + α · ema[i] / median(ema), 1/clip, clip)`.
4. Loss in the minibatch is `mean(weight_b · mean_j (pred_bj − Y_bj)²)`.
5. Val loss for early stopping is unweighted MSE so the early-stop
   criterion stays comparable across modes.

## Results

| metric                | uniform (n=3)  | emae (n=3)     | delta   |
|-----------------------|---------------:|---------------:|--------:|
| val mean overhead %   | 1.993 ± 0.246  | 2.177 ± 0.307  | **+0.183 pp** |
| val argmin_acc %      | 59.37  ± 1.96  | 55.67  ± 3.63  | **−3.70 pp**  |
| train mean overhead % | 1.490 ± 0.203  | 1.720 ± 0.295  | +0.230 pp |
| train argmin_acc %    | 61.47  ± 1.63  | 58.03  ± 3.22  | −3.43 pp  |
| n_iter (early-stop)   | 345.7 ± 144.0  | 304.7 ± 114.0  | −41 epochs |
| final val loss        | 0.0162 ± 0.002 | 0.0169 ± 0.001 | +0.0007  |

### Per-seed table

| seed | mode    | val_overhead% | val_acc% | train_overhead% | train_acc% | n_iter |
|------|---------|--------------:|---------:|----------------:|-----------:|-------:|
| 1    | uniform | 2.25          | 60.5     | 1.45            | 61.8       | 322    |
| 2    | uniform | 1.76          | 60.5     | 1.31            | 62.9       | 500    |
| 3    | uniform | 1.97          | 57.1     | 1.71            | 59.7       | 215    |
| 1    | emae    | 2.25          | 57.4     | 1.66            | 58.8       | 430    |
| 2    | emae    | 1.84          | 58.1     | 1.46            | 60.8       | 207    |
| 3    | emae    | 2.44          | 51.5     | 2.04            | 54.5       | 277    |

Raw stderr logs in `benchmarks/hew_smoke/{uniform,emae}_seed{1,2,3}.stderr`.

## Reading the numbers

- The val-overhead regression (+0.18 pp) is smaller than the per-mode
  standard deviation (±0.25–0.31 pp), so the per-seed pairs don't have
  a tight enough mean for that delta on its own to fail the noise
  threshold.
- The val-argmin_acc regression (−3.70 pp) is **bigger than the
  combined seed-to-seed std** (1.96 + 3.63), and the same direction is
  reproduced on the train set (−3.43 pp). That's not seed noise.
- Crucially, `emae` **increased** seed-to-seed variance on argmin_acc
  (1.96 pp → 3.63 pp). Distill training was already noisy across seeds;
  feeding the loss with self-paced per-row weights compounds the
  noise.
- All 6 runs converged (n_iter ranged 207–500); early-stop didn't shorten
  emae runs in a way that suggests they died mid-improvement.

## Why it likely fails here

The picker distill loss is already heavily mass-imbalanced — the bytes
head has a wide log range (~10..14) and the scalar heads were
per-head-σ-normalized just before training to balance gradient
contribution. That per-head normalization is the right inverse-loss
weighting for the picker problem. Stacking per-row reweighting on top
biases capacity toward whichever (image, size, zq) rows happen to be
hardest **in a basis that doesn't match the codec decision boundary** —
a row can have a huge teacher-vs-student byte_log gap even though the
student's argmin pick is fine, and reweighting drags the network toward
fitting that gap at the expense of cleaner decision rows.

The technique probably has a place when:
- the target IS the metric you optimize against (i.e., regression
  loss correlates with downstream argmin), and
- the dataset has genuine long-tail hard examples that the model just
  isn't seeing enough.

Neither applies to the zenjpeg distill setup as it stands. The
HistGB→MLP distillation already smooths the teacher targets; the
"hard rows" the EMA flags are mostly teacher-noise outliers, not
under-represented hard content.

## Default

`--hard-example-weighting` defaults to **`none`** in
`train_hybrid.py`. The flag is wired through, the implementation is
correct (uniform-weight runs match the baseline behaviour to float
roundoff, verified by the matched seed-1 final losses 0.0163 / 0.0153
where the only difference is the weighted path with first-5-epoch
uniform warmup actually running but contributing nothing distinctive
in the warmup window), and codec configs that want to experiment can
opt in.

## Reproduce

```bash
cd /home/lilith/work/zen/zenjpeg
for mode in none emae; do
  for seed in 1 2 3; do
    PYTHONPATH=<zenanalyze>/zentrain/examples \
      python3 <zenanalyze>/zentrain/tools/train_hybrid.py \
        --codec-config zenjpeg_picker_config \
        --activation leakyrelu \
        --hard-example-weighting $mode \
        --seed $seed \
        --out-json <out>/${mode}_seed${seed}.json \
        --out-log <out>/${mode}_seed${seed}.log \
        > <out>/${mode}_seed${seed}.stderr 2>&1
  done
done
```

Each run takes ≈ 1–2 minutes wall on a single core (PyTorch
`set_num_threads(1)` is forced inside the student).
