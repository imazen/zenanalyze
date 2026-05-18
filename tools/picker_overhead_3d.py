#!/usr/bin/env python3
"""Picker overhead audit — bytes / encode_ms / zensim deltas vs oracle.

Existing train_multi_codec val metrics only report BYTES overhead.
The picker target IS bytes, so that's the primary signal. But the
encoder still has to RUN with the picked config — different configs
have different encode times AND can deliver different actual zensim
even at the same target zq (e.g. 4:4:4 vs 4:2:0 chroma).

This tool fills the gap. Per (image, target_zq) row in val:
  - picker pick   = argmin(predicted_bytes) over reachable+allowed cells
  - oracle pick   = argmin(actual_bytes)
  - delta_bytes   = bytes(picker) - bytes(oracle)           (always >= 0)
  - delta_time_ms = encode_ms(picker) - encode_ms(oracle)   (sign varies)
  - delta_zensim  = zensim(picker) - zensim(oracle)         (sign varies)

Reports mean / p50 / p75 / p90 / p95 / p99 / max for each delta both
as absolute and as % of oracle.

Usage:
  python3 tools/picker_overhead_3d.py <codec_config.py> <data_root>

Example:
  python3 tools/picker_overhead_3d.py \
    zentrain/examples/zenjpeg_picker_config.py /home/lilith/work/zen/zenjpeg
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "zentrain" / "tools"))
sys.path.insert(0, str(REPO_ROOT / "zentrain" / "examples"))

import train_hybrid as TH
from train_multi_codec import CodecConfigBundle, _bind_globals_for_codec


def summarize(name: str, deltas: np.ndarray, absolute: bool = False) -> dict:
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
        "min": float(finite.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_py")
    ap.add_argument("data_root")
    ap.add_argument("--picker-json", help="trained picker .json (joint trunk + head); "
                    "if omitted, falls back to in-sample teacher (HistGB-per-cell)")
    ap.add_argument("--codec-name", help="override codec name (default = config filename stem)")
    args = ap.parse_args()

    cfg_path = Path(args.config_py).resolve()
    codec = args.codec_name or cfg_path.stem.replace("_picker_config", "")
    bundle = CodecConfigBundle(codec_name=codec, config_path=cfg_path,
                                data_root=Path(args.data_root))
    print(f"\n=== {codec} — picker overhead audit ===")
    print(f"  config:   {cfg_path}")
    print(f"  pareto:   {bundle.pareto_path}")
    print(f"  features: {bundle.features_path}\n")

    _bind_globals_for_codec(bundle)
    pareto, ceilings, has_ceil, has_time = TH.load_pareto(bundle.pareto_path)
    feats, feat_cols, feat_transforms = TH.load_features(bundle.features_path)
    cells, cell_id_by_key, config_to_cell, parsed_all = TH.build_cell_index()
    n_cells = len(cells)
    print(f"  {len(pareto)} pareto keys, {len(feat_cols)} feats, {n_cells} cells")

    (
        Xs, Xe, bytes_log, scalars, reach, meta,
        time_log, metric_log, infeasible,
    ) = TH.build_dataset(
        pareto, feats, feat_cols, cells, config_to_cell, parsed_all,
        ceilings=ceilings if has_ceil else None,
        time_budget_multiplier=0.0,
        time_baselines=None,
        emit_metric_head=True,  # need metric_log too
        safety_default_cell_idx=None,
    )
    print(f"  decision rows: {len(Xs)}  (n_cells={bytes_log.shape[1]})")

    # Validation split (image-level holdout, seed 0xCAFE — same as trainer).
    rng = np.random.default_rng(0xCAFE)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * 0.20))
    val_set = set(images[:n_val])
    val_idx = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    print(f"  val rows: {len(val_idx)} / {len(meta)} ({len(val_set)} images)")

    bl = bytes_log[val_idx]
    rch = reach[val_idx]
    tl = time_log[val_idx] if time_log is not None else None
    ml = metric_log[val_idx] if metric_log is not None else None

    # For now: use the in-sample HistGB teacher as the picker (sanity
    # check the harness; replace with the JSON-trained MLP if requested).
    if args.picker_json:
        pj = json.loads(Path(args.picker_json).read_text())
        # The joint trained JSON predicts (n_cells + scalar_blocks) columns
        # using a 139-dim union input vector — we don't have that built
        # outside the trainer. Skip the JSON path for now and use the
        # teacher (which gets bytes_log directly).
        print(f"  WARN: --picker-json input requires the joint union-input "
              f"vector built by train_multi_codec; not implemented in this "
              f"standalone tool. Falling back to teacher.")
        args.picker_json = None

    # Use the teacher: train HistGB on train_idx, predict on val_idx.
    train_idx = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    scalars_tr = {ax: scalars[ax][train_idx] for ax in scalars}
    print(f"  fitting in-sample HistGB-per-cell teacher on {len(train_idx)} rows...")
    teachers_bytes, _t_per_axis, _scalar_means, _, _, _, _ = TH.train_teacher_per_cell(
        Xs[train_idx], bytes_log[train_idx], scalars_tr,
        reach[train_idx], n_cells,
        params=None,  # full HistGB
        time_log_tr=None,
        metric_log_tr=None,
    )
    pred_bl = np.column_stack([teachers_bytes[c].predict(Xs[val_idx]) for c in range(n_cells)])

    # Per-row picker pick vs oracle pick + 3D deltas
    n_val = bl.shape[0]
    bytes_pct = np.full(n_val, np.nan)
    time_diff_ms = np.full(n_val, np.nan)
    time_pct = np.full(n_val, np.nan)
    zensim_diff = np.full(n_val, np.nan)
    zensim_pct = np.full(n_val, np.nan)
    correct = 0
    n_evaluated = 0
    for i in range(n_val):
        m = rch[i]
        if not np.any(m):
            continue
        actual_b = np.where(m, np.exp(bl[i]), np.inf)
        pred_b = np.where(m, np.exp(np.clip(pred_bl[i], -30, 30)), np.inf)
        oracle = int(np.argmin(actual_b))
        pick = int(np.argmin(pred_b))
        bytes_pct[i] = (actual_b[pick] - actual_b[oracle]) / actual_b[oracle]
        if tl is not None:
            t_pick = float(np.exp(tl[i, pick])) if np.isfinite(tl[i, pick]) else np.nan
            t_oracle = float(np.exp(tl[i, oracle])) if np.isfinite(tl[i, oracle]) else np.nan
            if np.isfinite(t_pick) and np.isfinite(t_oracle) and t_oracle > 0:
                time_diff_ms[i] = t_pick - t_oracle
                time_pct[i] = (t_pick - t_oracle) / t_oracle
        if ml is not None:
            # metric_log is log(zensim) per cell; convert back to linear
            z_pick = float(np.exp(ml[i, pick])) if np.isfinite(ml[i, pick]) else np.nan
            z_oracle = float(np.exp(ml[i, oracle])) if np.isfinite(ml[i, oracle]) else np.nan
            if np.isfinite(z_pick) and np.isfinite(z_oracle):
                zensim_diff[i] = z_pick - z_oracle
                if z_oracle != 0.0:
                    zensim_pct[i] = (z_pick - z_oracle) / abs(z_oracle)
        if pick == oracle:
            correct += 1
        n_evaluated += 1

    print(f"\n  n_evaluated: {n_evaluated}")
    print(f"  argmin_acc:  {100 * correct / n_evaluated:.2f}%  (picker==oracle)")
    print(f"  miss rate:   {100 * (n_evaluated - correct) / n_evaluated:.2f}%")

    # Pretty-print summaries
    def show(label, values, scale=100.0, unit="%", fmt=":7.3f"):
        s = summarize(label, values)
        if s["n"] == 0:
            print(f"  {label:20s}: no finite samples")
            return
        ks = ["mean", "p50", "p75", "p90", "p95", "p99", "max"]
        parts = " ".join(f"{k}={s[k]*scale:7.3f}{unit}" for k in ks)
        print(f"  {label:20s}: {parts}")

    print("\n  --- BYTES overhead (picked vs oracle, % over oracle bytes) ---")
    show("bytes_overhead_%", bytes_pct, scale=100.0, unit="%")

    print("\n  --- ENCODE TIME delta (positive = slower than oracle) ---")
    show("time_delta_ms", time_diff_ms, scale=1.0, unit="ms")
    show("time_delta_%", time_pct, scale=100.0, unit="%")

    print("\n  --- ZENSIM delta (linear units, sign-preserved) ---")
    show("zensim_delta",   zensim_diff, scale=1.0, unit="")
    show("zensim_delta_%", zensim_pct, scale=100.0, unit="%")


if __name__ == "__main__":
    main()
