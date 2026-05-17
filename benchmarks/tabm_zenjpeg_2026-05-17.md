# TabM-style student A/B vs HistGB teacher — zenjpeg, 2026-05-17

**Question:** does a TabM-mini (BatchEnsemble) student beat the shipped
sklearn-MLP / PyTorch-LeakyReLU student on the zenjpeg picker
workload? The 2026-05-17 zentrain review proposed it as the "biggest
accuracy upside in the small-MLP family" — backed by the TabM paper
(ICLR 2025, arXiv:2410.24210) reporting +~2 pp average win across
46 tabular datasets.

**Answer:** no, by a wide margin, on this workload. The shipped
HistGB teacher remains the right baseline; TabM-as-student is a
measured regression.

## Reproduce

```bash
cd /home/lilith/work/zen/zenjpeg
PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \
    python3 <zenanalyze>/zentrain/tools/tabm_student_experiment.py \
        --codec-config zenjpeg_picker_config \
        --heads 8 --hidden 192,192,192 --epochs 150 \
        --out-json benchmarks/tabm_zenjpeg_2026-05-17.json
```

Data: `zq_pareto_2026-04-29.tsv` (3.5 M rows / 1388 keys) +
`zq_pareto_features_2026-05-01_parallel.tsv`. 12 cells, hybrid heads,
v2.1 35-feature schema. Image-level 80/20 train/val split, seed 0xCAFE.

## Results (held-out images)

| Model | argmin_acc | mean overhead | p99 overhead | fit time |
|---|--:|--:|--:|--:|
| **Teacher** (HistGB, joblib parallel) | **56.5%** | **1.81%** | (not reported) | (CPU, ~30s) |
| **TabM-mini ensemble** (K=8, 192×192×192, 150 epochs) | 42.2% | 7.66% | (not reported) | 3.8 s on RTX 5070 |
| TabM per-head mean | 29.6% (±2.6%) | (not aggregated) | — | — |

Ensemble averaging gave a clean +12.6 pp lift over per-head argmin acc
(29.6% → 42.2%) — TabM-mini does what the paper claims architecturally.
But the ensembled student still **loses to the teacher by 14.3 pp**
and **4.2× higher mean overhead**.

For context: the shipped v2.1 baseline MLP student (zenjpeg_picker_v2.1
docs) hits ~52% argmin acc on the same held-out split. **TabM also
loses to the baseline MLP by ~10 pp**, so this isn't a "distill noise"
artifact — TabM is structurally worse here.

## Why TabM loses on this workload

The picker's task is fundamentally **argmin selection over discrete
cells**, not regression accuracy per cell. The HistGB teacher's tree
structure captures categorical-discrimination decision boundaries
that an MLP — single OR ensembled — must learn from soft targets +
MSE, a much harder gradient signal. A few notes:

1. **MSE on bytes_log doesn't optimize argmin.** Two cells with
   bytes_log predictions of `(10.1, 10.2)` vs `(9.9, 10.0)` have the
   same MSE-from-actual but very different argmin behavior. The
   teacher learned the categorical ranking implicitly; the MLP/TabM
   chases MSE.

2. **K=8 heads × 150 epochs plateaued at val_mse ≈ 0.49** by epoch 100
   (see trace below). Running longer doesn't help — capacity is not
   the limit, it's the optimization target.

3. **The TabM paper benchmarks against generic tabular regression
   tasks.** Picker training is closer to a *ranking* problem where
   GBDTs traditionally outperform deep nets — the +2 pp upside the
   paper reports doesn't generalize when the task has structure
   GBDTs handle natively.

## What would actually beat the teacher on argmin

Not measured here; future work:

- **Listwise / ranknet-style loss** over the per-row cell vector
  (force the student to match the teacher's ranking, not its scalar
  predictions). Different objective entirely.
- **MLP + pair-margin auxiliary loss** — penalize cases where the
  student flips the teacher's pairwise cell ordering.
- **Co-teaching: train the MLP on the teacher's argmin labels
  directly** (hard target) plus the soft bytes_log target, weighted.
- **A model that doesn't try to win on accuracy at all** — accept
  that the student is the runtime constraint, ship the teacher's
  argmin via a small lookup table per-feature-bin (cheap categorical
  predictor with no neural training).

For now: keep the shipped HistGB → distilled-MLP pipeline. Don't
ship TabM.

## Per-epoch trace (for completeness)

```
[tabm] epoch   2  tr_loss 12.3633  val_mse 4.5193  lr 9.99e-04
[tabm] epoch  10  tr_loss  2.0553  val_mse 1.8078  lr 9.87e-04
[tabm] epoch  50  tr_loss  0.7278  val_mse 0.6549  lr 7.41e-04
[tabm] epoch 100  tr_loss  0.5390  val_mse 0.5003  lr 2.41e-04
[tabm] epoch 140  tr_loss  0.5218  val_mse 0.4866  lr 8.86e-06
```

Cosine LR schedule from `1e-3 → 0` over 150 epochs. The val curve
is monotone decreasing, no overfitting — purely capacity / objective
ceiling.

## Implication for the prior recommendation

The 2026-05-17 zentrain review (this session) proposed TabM as the
biggest accuracy upside in the small-MLP family. That guess was
wrong for this workload's task shape. The prototype script lives at
`zentrain/tools/tabm_student_experiment.py` and stays in-tree so
future investigators can re-evaluate on a different task structure
(e.g., the metric scorer in zensim, where pure regression accuracy
is the win condition — TabM may pay off there).

## Sources

- TabM paper: [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
- BatchEnsemble: [arXiv:2002.06715](https://arxiv.org/abs/2002.06715)
- TabM official implementation: [yandex-research/tabm](https://github.com/yandex-research/tabm)
- Prototype: [`zentrain/tools/tabm_student_experiment.py`](../zentrain/tools/tabm_student_experiment.py)
- Raw JSON: [`tabm_zenjpeg_2026-05-17.json`](tabm_zenjpeg_2026-05-17.json)
