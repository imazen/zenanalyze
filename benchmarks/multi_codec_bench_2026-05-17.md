# Multi-codec shared-trunk MLP — 2026-05-17 bake

Status: prototype landed in `zentrain/tools/train_multi_codec.py`,
smoke-tested on real data for zenjpeg + zenwebp (+ zenavif). Negative
caveats documented at the bottom — DO NOT bake any of these JSONs
into a production runtime without first wiring the union-feature
adapter described below.

## What this is

A single shared-trunk MLP trained simultaneously on multiple codecs'
Pareto data. The trunk is a codec-agnostic image-feature → hidden
representation. Each codec gets its own final-layer Linear head
(different `n_cells`, different scalar axes per codec).

Per-codec teachers run the existing single-codec pipeline
(`load_pareto` / `load_features` / `build_dataset` /
`train_teacher_per_cell`) and emit soft targets exactly like
`train_hybrid.py` does. The student is the only thing that changes —
one shared-trunk PyTorch model instead of N independent
`MLPRegressor`s.

## Architecture

```
Input layout (per row, n_inputs = 2*n_union + 4 + 2 + n_codecs):
  [union_feats (n_union)]      raw value where codec uses it, 0 elsewhere
  [presence_mask (n_union)]    1.0 where codec uses it, 0 elsewhere
  [size_tiny, size_small, size_medium, size_large]
  [log_pixels, zq_norm]
  [codec_onehot (n_codecs)]

Trunk = Linear(n_inputs, h0) -> LeakyReLU -> Linear(h0, h1) -> LeakyReLU
Per-codec head = Linear(h1, (1 + len(SCALAR_AXES)) * n_cells)
```

Codec-balanced sampling: each step pulls `batch_size` rows from
EACH codec independently, so the trunk sees a roughly equal mix
regardless of relative dataset size.

**Per-codec loss normalization (mandatory, learned the hard way):**
without dividing each codec's MSE by its own label-variance, the
gradient is dominated by whichever codec has the widest log-bytes
range — and that codec's high-zq tail bands hold all the signal,
crushing the rest. Variance-normalized loss is the diff between
the v0 (broken) and v1 (working) runs below.

## Measured deltas (2-codec run: zenjpeg + zenwebp)

| Codec   | Run      | Argmin acc | Mean overhead | p90 overhead | Notes |
|---------|----------|-----------:|--------------:|-------------:|-------|
| zenjpeg | baseline single-codec  | 48.3%      | 3.66%         | -            | v2.2-clean, 51 feats, MLP 112→128→128→36 |
| zenjpeg | shared-trunk 2-codec   | **51.3%**  | **2.77%**     | 7.98%        | trunk 128, head 64, hidden 128x64 |
| zenjpeg | shared-trunk 3-codec   | 43.5%      | 3.62%         | 10.16%       | adding zenavif hurts zenjpeg |
| zenwebp | baseline single-codec  | 13.9%      | 3.97%         | -            | v0.2, 78 feats, MLP 78→192³→72 (much bigger model) |
| zenwebp | shared-trunk 2-codec   | 34.2%      | 2.74%         | 7.27%        | smaller model, **+20pp acc** |
| zenwebp | shared-trunk 3-codec   | **53.8%**  | **1.62%**     | 5.24%        | adding zenavif helps zenwebp dramatically |
| zenavif | baseline single-codec  | 23.3%      | 5.60%         | -            | 114→128²→20 |
| zenavif | shared-trunk 3-codec   | 21.6%      | **3.92%**     | 13.13%       | slight acc drop but -1.68pp overhead |

### Read-outs

- **zenjpeg 2-codec: +3pp acc, -0.89pp overhead** vs baseline. A clear
  win for the data-rich codec — the shared trunk sees zenwebp's
  signal and improves rather than degrading. The 3-codec version
  loses 5pp of acc relative to 2-codec; adding zenavif (smallest
  dataset, no scalar heads) regularizes the trunk in a direction
  that doesn't help zenjpeg.
- **zenwebp 2-codec: +20pp acc**, and 3-codec: **+40pp acc** vs the
  bigger baseline model. The shipped baseline is partially
  data-starved (only 247 images contributed to the v0.2 sweep).
  Pooling with zenjpeg + zenavif's images and feature signal gives
  the trunk a much richer view of the image manifold than zenwebp's
  rows alone would. This is the expected multi-task tabular win
  for data-poor codecs.
- **zenavif 3-codec: -1.68pp overhead** despite -1.7pp acc. The
  picker's argmin is slightly less accurate but its mis-picks are
  closer to the optimum (lower overhead-when-wrong). Probably
  because the shared trunk smooths out small noise in zenavif's
  bytes-log predictions.

### Caveats and known issues

1. **Bake compatibility is not yet runtime-compatible.** The emitted
   JSON has `feat_cols = union_features` and an `extra_axes` list
   covering the trunk's full input layout (presence mask + size +
   log_pixels + zq_norm + codec onehot). `bake_picker.py` accepts
   that shape, but the codec runtime currently reconstructs the
   input vector from its own KEEP_FEATURES — it would have to:
   - know the trunk's union vocabulary (which features to fill, which
     to leave at zero)
   - emit a presence-mask channel parallel to the union vector
   - emit a codec-onehot at inference (which slot is "this codec")
   - know that engineered axes log_pixels + zq_norm aren't repeated
     (the shared trunk uses only the simple form, not the
     `log_pixels^2 + zq_norm*feat` cross-terms `train_hybrid` uses)

   Until that adapter is wired into a codec runtime, these bakes
   are research artifacts — they exercise the training path and
   prove the architecture works, but they don't drop into a
   shipping decoder.

2. **`feature_transforms` is `identity` for all union slots.**
   Per-codec FEATURE_TRANSFORMS (log / log1p on heavy-tailed
   features) are applied INSIDE `load_features` per codec, then the
   transformed values land in the trunk input. A future runtime
   adapter would need to apply the transform per-feature per-codec
   before populating the union vector. Punting this for now keeps
   the JSON schema flat.

3. **3-way zenjpeg regression.** Adding zenavif pushed zenjpeg from
   51.3% → 43.5% argmin acc. Hypotheses:
   - zenavif's tiny training set (5205 rows vs zenjpeg's 23348)
     contributes only ~14% of optimizer steps but its
     variance-normalized gradient signal isn't tiny.
   - zenavif uses a different metric direction interpretation
     internally (the metric is still zensim, but the rav1e config
     names use a different scale).
   - Variance-normalized loss may need a soft floor — when a
     codec's loss is dominated by easy bytes_log (no scalar heads),
     its `loss_scale` is small and its effective gradient weight is
     unbounded.

   The fix is straightforward (clamp `loss_scale` floor, or use
   number of outputs as a normalizer instead of variance). Not done
   in this prototype.

4. **Early stop fires too soon at large epoch budgets.** Both 2-way
   and 3-way training stop around ep 80-110 because the
   variance-normalized aggregate plateaus while individual codecs
   continue to improve. A per-codec early-stop signal (or a
   per-codec head freeze when its val loss stops moving) would
   keep the trunk training longer for the codecs that still
   benefit.

## Files

- `zentrain/tools/train_multi_codec.py` — the trainer.
- `benchmarks/multi_codec_2026-05-17/multi_codec_zenjpeg_picker.json`
  (2-way) and `_3way_*` (3-way) — per-codec bakes in the JSON
  shape `bake_picker.py` consumes.
- `benchmarks/multi_codec_2026-05-17/multi_codec_summary.json` and
  `multi_codec_3way_summary.json` — training metadata + per-codec
  metrics in machine-readable form.

## Reproduction

From the zenanalyze repo root (or any worktree thereof):

    python3 zentrain/tools/train_multi_codec.py \
      --codec zenjpeg=zentrain/examples/zenjpeg_picker_config.py:/home/lilith/work/zen/zenjpeg \
      --codec zenwebp=zentrain/examples/zenwebp_picker_config.py:/home/lilith/work/zen/zenwebp \
      --out-dir benchmarks/multi_codec_2026-05-17 \
      --hidden 128,64 \
      --epochs 300

Wall time on this hardware: ~30 s total for 2 codecs (cached
HistGB teachers; cold-cache is ~120 s including TSV parsing).

## Next chunks (queued for the integrator)

- Add a `--head-init xavier_normal_scaled` option so per-codec
  heads don't all start near-identical.
- Wire a runtime-side union-vector adapter in `zenpicker` (or a
  new `zenanalyze::multi_codec_input`) so these bakes can actually
  ship.
- Sweep `--hidden` (try 96x48x32 and 192x96) — current 128x64 was
  picked from a guess, not a sweep.
- Add the missing codec (zenjxl); the features TSV at
  `/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_lossy_features_2026-05-01.tsv`
  was missing at run time. Likely needs `refresh_features.py` against
  the lossy sweep.
- Decide: ship as the production picker, or keep as a regularizer
  during single-codec training (use the shared trunk's hidden
  representation as an additional feature to the single-codec
  picker)? Both are valid; the data-poor codec wins suggest the
  former for new codecs and the latter for established ones.
