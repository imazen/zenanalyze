#!/usr/bin/env bash
# Retrain + eval the zenjpeg picker on a CV-chosen top-K feature subset,
# via the SAME pipeline the shipped 108-feature bake used.
#
# Prereqs (produced by cv_topk.py): cv_ranking.npz in $CKPT.
# Usage: run_topk_retrain.sh <K>
set -euo pipefail

K="${1:?usage: run_topk_retrain.sh <K>}"
CKPT=/mnt/v/zen/picker-retrain-top40-2026-06-01
REPO=/home/lilith/work/zen/zenanalyze/zenpicker-train
BIN=/home/lilith/work/zen/zenanalyze/target/release/zenpicker-train
ZENPREDICT=/home/lilith/work/zen/zenanalyze/target/release/zenpredict
FIXED=/mnt/v/zen/picker-dense-full-2026-05-27/parquet/picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet
IN_FO=/mnt/v/zen/picker-dense-full-2026-05-27/parquet/feature_order.txt

OUTDIR="$CKPT/top${K}"
mkdir -p "$OUTDIR" "$CKPT/logs"
# Stage bake bytes on LOCAL disk first; /mnt/v is a flaky WSL drive that
# occasionally drops just-written small files. Copy to $OUTDIR at the end.
LOCAL_OUT="$HOME/.cache/picker-retrain-stage/top${K}"
rm -rf "$LOCAL_OUT"; mkdir -p "$LOCAL_OUT"
BASE="picker_zenjpeg_A_top${K}_none_v3"

# 1. Rebuild the training parquet with only the top-K source features.
echo "[run] building top-${K} parquet ..."
python3 "$REPO/scripts/ablation/build_topk_parquet.py" \
  --in-parquet "$FIXED" \
  --ranking-npz "$CKPT/cv_ranking.npz" \
  --in-feature-order "$IN_FO" \
  --k "$K" \
  --out-parquet "$OUTDIR/picker_zenjpeg_A_top${K}.parquet" \
  --out-feature-order "$OUTDIR/feature_order.txt" \
  2>&1 | tee "$CKPT/logs/build_top${K}_parquet.log"

# 2. Retrain via the SAME pipeline the 108-bake used: --distill with the
#    DEFAULT bounded grid search (NO --hidden/--seed). The 108-bake's
#    manifest shows kind="bounded_grid", selection_metric="heldout_argmin_
#    accuracy"; it selected candidate #1 ([64,64], lr=1e-3, seed=0). Let
#    the grid select for the top-K input exactly the same way.
echo "[run] retraining (distill, default bounded grid, argmin selection) ..."
"$BIN" \
  --input "$OUTDIR/picker_zenjpeg_A_top${K}.parquet" \
  --codec zenjpeg \
  --out "$LOCAL_OUT/${BASE}.bin" \
  --distill \
  --val-frac 0.2 \
  2>&1 | tee "$CKPT/logs/train_top${K}.log"

# 3. Pack to f16 (the shipped form), same flags the 108-bake used.
echo "[run] packing f16 ..."
"$ZENPREDICT" repack \
  "$LOCAL_OUT/${BASE}.bin" \
  "$LOCAL_OUT/${BASE}_f16.bin" \
  --dtype f16 --zerobias 0.005 --compress \
  2>&1 | tee "$CKPT/logs/repack_top${K}_f16.log"

# 4. Eval held-out: f32 + f16, from the LOCAL staged bakes, on the top-K
#    parquet's own (same-image) val split.
echo "[run] eval f32 ..."
"$BIN" --input "$OUTDIR/picker_zenjpeg_A_top${K}.parquet" --codec zenjpeg --val-frac 0.2 \
  --eval-bake "$LOCAL_OUT/${BASE}.bin" \
  2>&1 | tee "$CKPT/logs/eval_top${K}_f32.log"
echo "[run] eval f16 ..."
"$BIN" --input "$OUTDIR/picker_zenjpeg_A_top${K}.parquet" --codec zenjpeg --val-frac 0.2 \
  --eval-bake "$LOCAL_OUT/${BASE}_f16.bin" \
  2>&1 | tee "$CKPT/logs/eval_top${K}_f16.log"

# 5. Copy bakes (+ manifest) to the /mnt/v checkpoint dir and verify they land.
cp -f "$LOCAL_OUT/${BASE}.bin" "$OUTDIR/"
cp -f "$LOCAL_OUT/${BASE}_f16.bin" "$OUTDIR/"
cp -f "$LOCAL_OUT/${BASE}.bin.toml" "$OUTDIR/" 2>/dev/null || true
sync
echo "[run] DONE top-${K}; staged + checkpoint sha256s:"
sha256sum "$LOCAL_OUT/${BASE}.bin" "$LOCAL_OUT/${BASE}_f16.bin"
ls -la "$OUTDIR/${BASE}.bin" "$OUTDIR/${BASE}_f16.bin" || echo "WARN: /mnt/v copy not visible yet"
