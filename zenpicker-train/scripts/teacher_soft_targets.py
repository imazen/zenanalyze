#!/usr/bin/env python3
"""Generate the HistGB teacher's DENSE soft `bytes_log` targets for the
within-cell-optimal picker — the offline target-generation half of
zentrain's teacher → student distillation, faithfully replicated.

This script is a ONE-TIME, build-side step. The Rust `zenpicker-train`
runtime gains NO Python dependency: it exports the exact dataset it
built (raw teacher inputs + hard `bytes_log` + the `reach` mask + the
grouped train/val split, keyed by `row_idx`), shells to this script
once, and reads the soft targets back to distill the pure-Rust MLP
student.

Recipe (matches `zentrain/tools/train_hybrid.py`
`train_teacher_per_cell` + `teacher_predict_all`):

  for each categorical cell c in 0..n_cells:
      mask = train rows where reach_c (bytes_log_c is non-null)
      if mask.sum() < min_cell_rows:                       # zentrain: 50
          teacher_c = None        # student gets the per-cell nanmean
      else:
          teacher_c = HistGradientBoostingRegressor(**HISTGB_FULL)
          teacher_c.fit(X_train[mask], bytes_log_c_train[mask])
  # DENSE soft targets for ALL rows (train + val), no NaN holes:
  soft[:, c] = teacher_c.predict(X_all)  if teacher_c is not None
               else nanmean(bytes_log_c_train)

The student (in Rust) then trains on `soft[train_rows]` via pure MSE.
The held-out decision-quality metric (argmin vs the true oracle) is
computed in Rust against the HARD targets — distillation changes the
training target only. For provenance we ALSO compute the teacher's own
held-out argmin accuracy + mean byte overhead here (the distillation
ceiling) and emit it to `--stats-out`.

Inputs (the Rust `export_teacher_dataset` parquet):
  row_idx:i64, image_id:utf8, target_zq:i64, split:i64 (0=train,1=val),
  reach_{c}:i64, bytes_log_{c}:f32 (nullable), f_{j}:f32 (raw inputs).

Output (`--out` parquet):
  row_idx:i64, soft_{c}:f32  + KV metadata:
    source_export_sha256, n_cells_with_teacher, teacher_params.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, type=Path, help="teacher dataset parquet")
    ap.add_argument("--out", required=True, type=Path, help="soft-target parquet to write")
    ap.add_argument("--stats-out", type=Path, default=None, help="teacher held-out stats JSON")
    ap.add_argument("--n-cells", required=True, type=int)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--l2-regularization", type=float, default=0.5)
    ap.add_argument("--min-cell-rows", type=int, default=50)
    ap.add_argument("--random-state", type=int, default=0xCAFE)
    args = ap.parse_args()

    n_cells = args.n_cells
    table = pq.read_table(args.export)
    cols = table.column_names

    feat_cols = sorted(
        [c for c in cols if c.startswith("f_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not feat_cols:
        raise SystemExit("export parquet has no f_* feature columns")

    row_idx = table.column("row_idx").to_numpy(zero_copy_only=False).astype(np.int64)
    split = table.column("split").to_numpy(zero_copy_only=False).astype(np.int64)
    n_rows = len(row_idx)

    X = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in feat_cols]
    )

    # bytes_log_{c}: nullable f32 -> NaN where null. reach_{c}: 0/1.
    bytes_log = np.full((n_rows, n_cells), np.nan, dtype=np.float64)
    reach = np.zeros((n_rows, n_cells), dtype=bool)
    for c in range(n_cells):
        bl = table.column(f"bytes_log_{c}").to_numpy(zero_copy_only=False)
        bytes_log[:, c] = np.asarray(bl, dtype=np.float64)
        rc = table.column(f"reach_{c}").to_numpy(zero_copy_only=False).astype(np.int64)
        reach[:, c] = rc != 0

    is_train = split == 0
    is_val = split == 1
    X_tr = X[is_train]
    bl_tr = bytes_log[is_train]

    histgb = dict(
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        random_state=args.random_state,
    )
    sys.stderr.write(
        f"[teacher] {n_rows} rows ({is_train.sum()} train / {is_val.sum()} val), "
        f"{len(feat_cols)} inputs, {n_cells} cells; HISTGB={histgb} min_cell_rows={args.min_cell_rows}\n"
    )

    # Per-cell teacher fit on the reaching TRAIN rows (zentrain's mask).
    teachers: list[HistGradientBoostingRegressor | None] = [None] * n_cells
    fallback_mean = np.full(n_cells, 0.0, dtype=np.float64)
    n_with_teacher = 0
    for c in range(n_cells):
        y_col = bl_tr[:, c]
        mask = np.isfinite(y_col)
        # zentrain's per-cell nanmean fallback for under-sampled cells.
        fm = float(np.nanmean(y_col)) if mask.any() else 0.0
        fallback_mean[c] = fm if np.isfinite(fm) else 0.0
        if mask.sum() < args.min_cell_rows:
            teachers[c] = None
            continue
        gbm = HistGradientBoostingRegressor(**histgb)
        gbm.fit(X_tr[mask], y_col[mask])
        teachers[c] = gbm
        n_with_teacher += 1
    sys.stderr.write(f"[teacher] {n_with_teacher}/{n_cells} cells got a per-cell teacher\n")

    # DENSE soft targets for ALL rows (teacher_predict_all).
    soft = np.zeros((n_rows, n_cells), dtype=np.float32)
    for c in range(n_cells):
        if teachers[c] is None:
            soft[:, c] = fallback_mean[c]
        else:
            soft[:, c] = teachers[c].predict(X).astype(np.float32)

    # Teacher's OWN held-out argmin accuracy + mean byte overhead on val
    # (the distillation ceiling). Pure provenance; the Rust side recomputes
    # the student's numbers identically.
    def argmin_overhead(pred: np.ndarray, truth: np.ndarray, rch: np.ndarray):
        correct = 0
        scored = 0
        overheads = []
        for i in range(pred.shape[0]):
            reachable = np.where(rch[i])[0]
            if reachable.size == 0:
                continue
            scored += 1
            # masked argmin over predicted bytes_log
            pc = reachable[np.argmin(pred[i, reachable])]
            tc = reachable[np.argmin(truth[i, reachable])]
            if pc == tc:
                correct += 1
            best = float(np.exp(truth[i, tc]))
            pick = float(np.exp(truth[i, pc]))
            if best > 0 and np.isfinite(pick):
                overheads.append(pick / best - 1.0)
        argmin_acc = correct / scored if scored else 0.0
        mean_oh = float(np.mean(overheads)) if overheads else float("nan")
        return argmin_acc, mean_oh, scored

    t_argmin, t_overhead, t_scored = argmin_overhead(
        soft[is_val], bytes_log[is_val], reach[is_val]
    )
    sys.stderr.write(
        f"[teacher] held-out (val n={t_scored}): argmin_acc {t_argmin:.4f} "
        f"mean overhead {t_overhead:.4f}\n"
    )

    # Write soft targets keyed by row_idx + provenance metadata.
    arrays = {"row_idx": pa.array(row_idx, type=pa.int64())}
    for c in range(n_cells):
        arrays[f"soft_{c}"] = pa.array(soft[:, c], type=pa.float32())
    out_table = pa.table(arrays)
    meta = {
        b"source_export_sha256": sha256_file(args.export).encode(),
        b"n_cells_with_teacher": str(n_with_teacher).encode(),
        b"teacher_params": json.dumps(histgb).encode(),
        b"min_cell_rows": str(args.min_cell_rows).encode(),
        b"recipe": b"histgb_per_cell_soft_target_mse",
    }
    out_table = out_table.replace_schema_metadata(meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # zstd only: the Rust `parquet` crate is built with the `zstd` feature
    # (not snappy), so the default snappy codec would be unreadable there.
    pq.write_table(out_table, args.out, compression="zstd")
    sys.stderr.write(f"[teacher] wrote soft targets → {args.out}\n")

    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(
            json.dumps(
                {
                    "argmin_acc": t_argmin,
                    "overhead_mean": t_overhead,
                    "n_val_scored": t_scored,
                    "n_cells_with_teacher": n_with_teacher,
                    "n_cells": n_cells,
                    "teacher_params": histgb,
                },
                indent=2,
            )
        )
        sys.stderr.write(f"[teacher] wrote stats → {args.stats_out}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
