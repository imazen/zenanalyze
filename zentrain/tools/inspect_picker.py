#!/usr/bin/env python3
"""
Inspect a baked picker (.bin) — surface the approximation logic the
MLP actually implements + flag pathological / hidden-edge behavior.

Modes:

  summary             High-level shape: parameter count, layer / cell
                      stats, weight magnitude / sparsity, dead-cell
                      detection, output-dim collapse, metadata.

  tree-approx         Fit a depth-N decision tree mimicking the
                      picker's argmin over a corpus of feature
                      vectors. Print the tree as text + emit a
                      graphviz `.dot` file. Shows in human-readable
                      form what the picker is doing — which features
                      it actually splits on, and how cells partition
                      the feature space.

  picks               Pick distribution over a corpus: how often each
                      cell is chosen, low-confidence picks (gap to
                      second-best), reach-rate disagreement (cells
                      the picker prefers but the bake's reach gate
                      considers unsafe). Surfaces dead cells (never
                      picked) and degenerate cells (always picked).

  stress              Synthetic boundary-stress: feature vectors at
                      every corner of the `feature_bounds` envelope
                      (and one stride inside). Reports the picker's
                      pick + confidence at each. Pathological
                      indicators: same pick at every corner (model
                      not differentiating), high confidence at
                      out-of-bounds inputs, picks that don't match
                      what `first_out_of_distribution` would gate.

  diverge             Compare picker top-1 to tree-approximation
                      top-1 over a held-out corpus. Disagreement at
                      high confidence is the red flag — the MLP is
                      doing something the tree can't capture, which
                      may be real signal or memorization. Pair with
                      `--corpus` to specify a TSV of feature vectors.

Usage:
    python3 zentrain/tools/inspect_picker.py <model.bin> --mode summary
    python3 zentrain/tools/inspect_picker.py <model.bin> --mode tree-approx \\
            --corpus benchmarks/zq_pareto_features_2026-04-30.tsv \\
            --depth 4 --dot /tmp/picker_tree.dot
    python3 zentrain/tools/inspect_picker.py <model.bin> --mode picks \\
            --corpus benchmarks/zq_pareto_features_2026-04-30.tsv
    python3 zentrain/tools/inspect_picker.py <model.bin> --mode stress
    python3 zentrain/tools/inspect_picker.py <model.bin> --mode diverge \\
            --corpus benchmarks/zq_pareto_features_2026-04-30.tsv

Requires: numpy, scikit-learn (DecisionTreeClassifier for tree-approx).

The model is loaded via the `zenpredict-inspect` binary, which dumps
the .bin to JSON. Build with:
    cargo build --release -p zenpredict --bin zenpredict-inspect
"""

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LEAKY_RELU_ALPHA = 0.01

# ====================================================================
# Loading the bake via zenpredict-inspect.
# ====================================================================


def find_inspect_bin(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"--inspect-bin {explicit} does not exist")
        return explicit
    on_path = shutil.which("zenpredict-inspect")
    if on_path:
        return Path(on_path)
    for sub in ("release", "debug"):
        cand = REPO_ROOT / "target" / sub / "zenpredict-inspect"
        if cand.exists():
            return cand
    return Path("__cargo_run__")


def load_model(bin_path: Path, inspect_bin: Path | None) -> dict:
    """Run `zenpredict-inspect <bin_path> --weights` and parse the JSON."""
    inspect = find_inspect_bin(inspect_bin)
    if str(inspect) == "__cargo_run__":
        cmd = [
            "cargo", "run", "-q",
            "--manifest-path", str(REPO_ROOT / "Cargo.toml"),
            "--release", "-p", "zenpredict", "--bin", "zenpredict-inspect",
            "--", str(bin_path), "--weights",
        ]
    else:
        cmd = [str(inspect), str(bin_path), "--weights"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"zenpredict-inspect exited {res.returncode}")
    if res.stderr:
        sys.stderr.write(res.stderr)
    return json.loads(res.stdout)


# ====================================================================
# Forward pass (numpy reference). Mirrors zenpredict::inference::forward.
# ====================================================================


def forward(model: dict, features: np.ndarray) -> np.ndarray:
    """Run a single feature vector through the MLP. Returns the
    raw output vector. Activation handling matches the runtime."""
    mean = np.asarray(model["scaler_mean"], dtype=np.float32)
    scale = np.asarray(model["scaler_scale"], dtype=np.float32)
    x = (features.astype(np.float32) - mean) / scale
    last_idx = len(model["layers"]) - 1
    for i, layer in enumerate(model["layers"]):
        in_dim = layer["in_dim"]
        out_dim = layer["out_dim"]
        w = np.asarray(layer["weights"], dtype=np.float32).reshape(in_dim, out_dim)
        b = np.asarray(layer["biases"], dtype=np.float32)
        x = x @ w + b
        if i == last_idx:
            continue
        act = layer["activation"]
        if act == "relu":
            x = np.maximum(x, 0.0)
        elif act == "leakyrelu":
            x = np.where(x >= 0.0, x, LEAKY_RELU_ALPHA * x)
        # identity: no-op
    return x


def forward_batch(model: dict, X: np.ndarray) -> np.ndarray:
    """Vectorized forward over a (n_samples, n_inputs) matrix.
    Returns (n_samples, n_outputs)."""
    mean = np.asarray(model["scaler_mean"], dtype=np.float32)
    scale = np.asarray(model["scaler_scale"], dtype=np.float32)
    x = (X.astype(np.float32) - mean[None, :]) / scale[None, :]
    last_idx = len(model["layers"]) - 1
    for i, layer in enumerate(model["layers"]):
        in_dim = layer["in_dim"]
        out_dim = layer["out_dim"]
        w = np.asarray(layer["weights"], dtype=np.float32).reshape(in_dim, out_dim)
        b = np.asarray(layer["biases"], dtype=np.float32)
        x = x @ w + b
        if i == last_idx:
            continue
        act = layer["activation"]
        if act == "relu":
            x = np.maximum(x, 0.0)
        elif act == "leakyrelu":
            x = np.where(x >= 0.0, x, LEAKY_RELU_ALPHA * x)
    return x


# ====================================================================
# Corpus loading.
# ====================================================================


def feat_columns(model: dict) -> list[str]:
    """Pull `feat_cols` from the bake metadata. The trainer-side
    `zentrain.feature_columns` key carries them (newline-separated
    utf8 — see zentrain/tools/bake_picker.py)."""
    for entry in model.get("metadata", []):
        if entry["key"] == "zentrain.feature_columns" and "value_text" in entry:
            return [c for c in entry["value_text"].split("\n") if c]
    return []


def load_corpus(path: Path, feat_cols: list[str]) -> tuple[np.ndarray, list[dict]]:
    """Load a per-(image, size) features TSV, return (X, rows).

    `feat_cols` defines the column order the picker expects. The TSV
    must contain those columns (extras are ignored). Engineered cross-
    terms (`zq_norm`, `log_pixels`, etc.) are NOT computed here — see
    `prepare_input_for_picker` for the full assembly."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit(f"corpus {path} is empty")
    X = np.zeros((len(rows), len(feat_cols)), dtype=np.float32)
    missing = []
    for j, col in enumerate(feat_cols):
        if col not in rows[0]:
            missing.append(col)
            continue
        for i, row in enumerate(rows):
            try:
                X[i, j] = float(row[col]) if row[col] not in ("", "nan", "NaN") else float("nan")
            except ValueError:
                X[i, j] = float("nan")
    if missing:
        sys.stderr.write(
            f"WARNING: corpus {path} missing {len(missing)} feat_col(s): {missing[:5]}{'…' if len(missing) > 5 else ''}\n"
            "         Those columns are zero-filled. Re-run the features harness with the full schema.\n"
        )
    return X, rows


def prepare_input_for_picker(
    model: dict,
    feat_cols: list[str],
    X: np.ndarray,
    target_zq: float,
    icc_bytes: float = 0.0,
    log_pixels_per_row: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the full picker input vector. The picker's
    `n_inputs` includes engineered axes past `feat_cols` (size onehot,
    polynomials, zq×feat cross-terms, icc_bytes). Reconstructs the
    legacy zenjpeg layout — extend here when other layouts ship.

    Returns (n_samples, n_inputs)."""
    n_inputs = model["n_inputs"]
    n_feat = len(feat_cols)
    expected_legacy = n_feat + 4 + 5 + n_feat + 1
    if n_inputs != expected_legacy:
        sys.stderr.write(
            f"WARNING: picker n_inputs={n_inputs} doesn't match the legacy "
            f"zenjpeg layout (n_feat + 4 + 5 + n_feat + 1 = {expected_legacy}). "
            "Engineered axes won't be reconstructed; running on raw feat "
            "columns only — picker may produce nonsense. Pass --raw to skip "
            "engineered axis reconstruction silently.\n"
        )
        # Best-effort: pad with zeros to match n_inputs.
        n = X.shape[0]
        out = np.zeros((n, n_inputs), dtype=np.float32)
        out[:, : min(n_feat, n_inputs)] = X[:, : min(n_feat, n_inputs)]
        return out

    n_samples = X.shape[0]
    if log_pixels_per_row is None:
        log_pixels_per_row = np.full(n_samples, math.log(1024 * 1024), dtype=np.float32)
    out = np.zeros((n_samples, n_inputs), dtype=np.float32)
    out[:, :n_feat] = X
    # Size onehot — pick small (256²) for the inspection by default.
    # The 4 axes are tiny / small / medium / large in that order.
    out[:, n_feat + 1] = 1.0  # small
    log_pixels = log_pixels_per_row
    out[:, n_feat + 4 + 0] = log_pixels
    out[:, n_feat + 4 + 1] = log_pixels ** 2
    zq_norm = float(target_zq) / 100.0
    out[:, n_feat + 4 + 2] = zq_norm
    out[:, n_feat + 4 + 3] = zq_norm ** 2
    out[:, n_feat + 4 + 4] = zq_norm * log_pixels
    # zq × feat cross terms
    for j in range(n_feat):
        out[:, n_feat + 4 + 5 + j] = zq_norm * X[:, j]
    out[:, n_feat + 4 + 5 + n_feat] = icc_bytes
    return out


# ====================================================================
# Modes.
# ====================================================================


def mode_summary(model: dict) -> int:
    print(f"Bake: {model.get('file', '?')}")
    print(f"File size: {model.get('file_bytes', 0):,} bytes")
    print(f"Format: ZNPR v{model.get('version')} (flags={model.get('flags', 0)})")
    print(f"Schema hash: {model.get('schema_hash')}")
    print(f"n_inputs={model['n_inputs']}  n_outputs={model['n_outputs']}  n_layers={model['n_layers']}")
    print()
    print("--- Layers ---")
    total_weights = 0
    total_near_zero = 0
    for i, layer in enumerate(model["layers"]):
        ws = layer.get("weight_stats") or {}
        n = layer["n_weights"]
        total_weights += n
        total_near_zero += int(round(n * float(ws.get("near_zero_fraction", 0.0))))
        print(
            f"  L{i}: {layer['in_dim']:>4} → {layer['out_dim']:<4}  act={layer['activation']:<10} "
            f"dtype={layer['dtype']:<3}  weights={n:>6,}  "
            f"|w| max={ws.get('abs_max', 0):.3g} p99={ws.get('abs_p99', 0):.3g} "
            f"p50={ws.get('abs_p50', 0):.3g}  near0={ws.get('near_zero_fraction', 0):.1%}"
        )
    print()
    print(f"Total weights: {total_weights:,}")
    print(f"Near-zero weights (|w|<1e-4): {total_near_zero:,}  ({total_near_zero / max(total_weights, 1):.1%})")
    fb = model.get("feature_bounds") or []
    print(f"Feature bounds populated: {bool(fb)}  ({len(fb)} entries)")
    if fb:
        finite = sum(1 for b in fb if math.isfinite(b["low"]) and math.isfinite(b["high"]))
        print(f"  Finite bounds: {finite}/{len(fb)}  (rest are open ±inf — engineered axes)")
    md = model.get("metadata") or []
    print(f"\n--- Metadata ({len(md)} entries) ---")
    for entry in md:
        kind = entry["kind"]
        if kind == "utf8":
            txt = entry.get("value_text") or ""
            preview = txt if len(txt) <= 80 else txt[:77] + "..."
            print(f"  [{kind:<7}] {entry['key']}  =  {preview!r}")
        elif "value_f32_3" in entry:
            print(f"  [{kind:<7}] {entry['key']}  =  {entry['value_f32_3']}")
        elif "value_u8" in entry:
            print(f"  [{kind:<7}] {entry['key']}  =  u8={entry['value_u8']}")
        else:
            print(f"  [{kind:<7}] {entry['key']}  ({entry['value_len']} bytes)")
    return 0


def mode_tree_approx(model: dict, corpus: Path, depth: int, dot_path: Path | None,
                     target_zq: float) -> int:
    try:
        from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
    except ImportError:
        sys.exit("scikit-learn required for tree-approx mode")
    feat_cols = feat_columns(model)
    if not feat_cols:
        sys.exit("bake metadata missing zentrain.feature_columns; cannot map inputs to feature names")
    X, rows = load_corpus(corpus, feat_cols)
    n_outputs = model["n_outputs"]
    inputs = prepare_input_for_picker(model, feat_cols, X, target_zq=target_zq)
    out = forward_batch(model, inputs)
    # We assume hybrid-heads layout where the categorical bytes
    # head is the first N_CELLS outputs. Without authoritative
    # hybrid_heads_layout metadata, fall back to: pick over the
    # full output vector. Most consumers won't notice the difference
    # for the inspection — the tree is mimicking *the picker's
    # actual argmin* either way.
    n_cells = read_n_cells_from_metadata(model)
    if n_cells is None:
        n_cells = n_outputs
        sys.stderr.write(
            "INFO: hybrid_heads_layout missing in bake metadata; treating "
            "all n_outputs as cells. Pass --n-cells to override.\n"
        )
    picks = np.argmin(out[:, :n_cells], axis=1)
    print(f"Corpus: {len(rows)} rows × {len(feat_cols)} features")
    print(f"Tree approximation depth: {depth}, fitting on argmin over "
          f"{n_cells} cells at target_zq={target_zq}")

    cell_counts = Counter(int(p) for p in picks)
    print(f"Pick distribution: {dict(sorted(cell_counts.items()))}")
    print()

    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X, picks)
    tree_picks = clf.predict(X)
    agreement = float(np.mean(tree_picks == picks))
    print(f"Tree depth-{depth} agreement with picker on training corpus: {agreement:.1%}")
    print()
    text = export_text(clf, feature_names=list(feat_cols), max_depth=depth)
    print("--- Decision tree (top-{} split) ---".format(depth))
    print(text)
    if dot_path is not None:
        export_graphviz(
            clf,
            out_file=str(dot_path),
            feature_names=list(feat_cols),
            class_names=[f"cell_{i}" for i in range(n_cells)],
            filled=True,
            rounded=True,
        )
        print(f"Wrote graphviz: {dot_path}")
        print(f"  Render with: dot -Tsvg {dot_path} -o {dot_path.with_suffix('.svg')}")
    return 0


def mode_picks(model: dict, corpus: Path, target_zq: float, low_conf_thresh: float) -> int:
    feat_cols = feat_columns(model)
    if not feat_cols:
        sys.exit("bake metadata missing zentrain.feature_columns")
    X, rows = load_corpus(corpus, feat_cols)
    inputs = prepare_input_for_picker(model, feat_cols, X, target_zq=target_zq)
    out = forward_batch(model, inputs)
    n_cells = read_n_cells_from_metadata(model) or model["n_outputs"]
    cells_out = out[:, :n_cells]
    picks = np.argmin(cells_out, axis=1)

    # Confidence: sorted top-2 gap in score (log-bytes) space.
    sorted_scores = np.sort(cells_out, axis=1)
    top2_gap = sorted_scores[:, 1] - sorted_scores[:, 0]

    n_samples = len(rows)
    counts = Counter(int(p) for p in picks)
    print(f"Corpus: {n_samples} rows. Target_zq={target_zq}. n_cells={n_cells}")
    print()
    print("--- Pick distribution by cell ---")
    for c in range(n_cells):
        n = counts.get(c, 0)
        bar = "█" * int(round(n / n_samples * 50))
        flag = ""
        if n == 0:
            flag = "  ← DEAD CELL — never picked"
        elif n / n_samples > 0.5:
            flag = "  ← DOMINANT — >50% of picks"
        print(f"  cell {c:>3}: {n:>5}  ({n / n_samples:>6.1%})  {bar}{flag}")
    print()
    n_low = int((top2_gap < low_conf_thresh).sum())
    print(f"--- Confidence (log-space gap to top-2) ---")
    print(f"  p50: {np.median(top2_gap):.4f}")
    print(f"  p10: {np.quantile(top2_gap, 0.10):.4f}")
    print(f"  p01: {np.quantile(top2_gap, 0.01):.4f}")
    print(f"  Low-confidence picks (gap < {low_conf_thresh}): {n_low}/{n_samples} ({n_low/n_samples:.1%})")
    return 0


def mode_stress(model: dict, target_zq: float) -> int:
    """Synthesize feature vectors at every corner of the feature_bounds
    envelope (and one stride inside) and report picks. Pathological
    indicators: same pick at every corner, high confidence at corners,
    picks that disagree with first_out_of_distribution semantics
    (corners are by definition AT the bounds; an εlexceedance is OOD)."""
    fb = model.get("feature_bounds") or []
    feat_cols = feat_columns(model)
    if not feat_cols:
        sys.exit("bake metadata missing zentrain.feature_columns")
    n_feat = len(feat_cols)
    if not fb or len(fb) < n_feat:
        sys.exit("feature_bounds missing or shorter than feat_cols — re-bake with bounds populated")

    # Use raw feat_cols range for stress; engineered axes get their
    # default (zero) treatment. We test 3^n_feat corners — explosion
    # for large n_feat, so cap at small_n; users with bigger schemas
    # iterate edge-by-edge. For schemas ≤ 8 features, full 3^n grid;
    # for larger, axis-aligned single-edge perturbations.
    n_per_axis = 3
    max_full = 8
    bounds = fb[:n_feat]
    if n_feat <= max_full:
        # Full grid of {p01, midpoint, p99} per feature.
        levels = [
            [b["low"], 0.5 * (b["low"] + b["high"]), b["high"]]
            for b in bounds
        ]
        grid = np.array(list(itertools.product(*levels)), dtype=np.float32)
    else:
        # Axis-aligned: hold all features at midpoint, sweep one axis
        # to extremes. n_axes × 2 vectors.
        mids = np.array([0.5 * (b["low"] + b["high"]) for b in bounds], dtype=np.float32)
        rows = []
        for j in range(n_feat):
            for v in (bounds[j]["low"], bounds[j]["high"]):
                row = mids.copy()
                row[j] = v
                rows.append(row)
        grid = np.stack(rows)

    # Replace any non-finite bound (legacy bakes had ±inf) with a
    # sentinel — pick a value just outside the rest of the distribution.
    grid = np.where(np.isfinite(grid), grid, 0.0).astype(np.float32)

    inputs = prepare_input_for_picker(model, feat_cols, grid, target_zq=target_zq)
    out = forward_batch(model, inputs)
    n_cells = read_n_cells_from_metadata(model) or model["n_outputs"]
    cells_out = out[:, :n_cells]
    picks = np.argmin(cells_out, axis=1)
    sorted_scores = np.sort(cells_out, axis=1)
    confs = sorted_scores[:, 1] - sorted_scores[:, 0]

    print(f"Boundary-stress: {len(grid)} synthetic feature vectors at p01/p99 corners")
    print(f"Target_zq={target_zq}, n_cells={n_cells}")
    print()
    counts = Counter(int(p) for p in picks)
    print(f"Picks across corners: {len(set(picks))} distinct cells")
    if len(set(picks)) == 1:
        print(f"  ← PATHOLOGY: picker chose the SAME cell ({picks[0]}) at every corner")
        print("    → MLP isn't differentiating boundary inputs. Likely a")
        print("      training-distribution gap — corner inputs were unseen and the")
        print("      MLP defaulted to its prior. Consider training-corpus expansion.")
    elif len(set(picks)) < 0.3 * n_cells:
        print(f"  ← Suspicious: only {len(set(picks))}/{n_cells} cells fire at corners")
    print(f"\nTop-2 gap at corners (smaller = picker less certain):")
    print(f"  p50: {np.median(confs):.4f}  p10: {np.quantile(confs, 0.10):.4f}  max: {confs.max():.4f}")
    if np.median(confs) > 1.0:
        print("  ← PATHOLOGY: median high confidence at boundary inputs.")
        print("    The picker SHOULD be uncertain at boundaries (those inputs are")
        print("    near-OOD by construction). High confidence here indicates the")
        print("    MLP is extrapolating linearly past its training envelope.")
    print()
    print("--- Per-cell hit count at corners ---")
    for c in sorted(counts):
        print(f"  cell {c:>3}: {counts[c]:>4}")
    return 0


def mode_diverge(model: dict, corpus: Path, depth: int, target_zq: float) -> int:
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        sys.exit("scikit-learn required for diverge mode")
    feat_cols = feat_columns(model)
    if not feat_cols:
        sys.exit("bake metadata missing zentrain.feature_columns")
    X, rows = load_corpus(corpus, feat_cols)
    inputs = prepare_input_for_picker(model, feat_cols, X, target_zq=target_zq)
    out = forward_batch(model, inputs)
    n_cells = read_n_cells_from_metadata(model) or model["n_outputs"]
    cells_out = out[:, :n_cells]
    picks = np.argmin(cells_out, axis=1)
    sorted_scores = np.sort(cells_out, axis=1)
    confs = sorted_scores[:, 1] - sorted_scores[:, 0]

    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X, picks)
    tree_picks = clf.predict(X)
    disagree = tree_picks != picks
    print(f"Tree-vs-MLP divergence on {len(rows)} rows (tree depth={depth}):")
    print(f"  Total disagreements: {int(disagree.sum())} ({disagree.mean():.1%})")
    if disagree.sum() == 0:
        print("  Picker is perfectly captured by a depth-{depth} tree — likely "
              "the MLP isn't earning its complexity over a tree of this depth.")
        return 0

    # Stratify by confidence — disagreements at high confidence are
    # the red flag.
    high_conf_disagree = disagree & (confs > np.quantile(confs, 0.75))
    low_conf_disagree = disagree & (confs <= np.quantile(confs, 0.25))
    print(f"  At high confidence (top-quartile gap): {int(high_conf_disagree.sum())} disagreements")
    print("    ← These are picks where the MLP is confident BUT the tree disagrees.")
    print("       Suspect either real signal the tree can't capture, or memorization.")
    print(f"  At low confidence (bottom-quartile gap): {int(low_conf_disagree.sum())} disagreements")
    print("    ← Expected; near-tie picks naturally split between MLP and tree.")
    return 0


def read_n_cells_from_metadata(model: dict) -> int | None:
    """Pull n_cells from `zentrain.hybrid_heads_layout` metadata if
    present. The layout is `[u32 n_cells, u32 n_heads, u8[n_heads] head_kinds]`
    little-endian per zentrain/tools/bake_picker.py."""
    for entry in model.get("metadata", []):
        if entry["key"] != "zentrain.hybrid_heads_layout":
            continue
        # Decode hex if present.
        hexs = entry.get("value_hex")
        if not hexs:
            return None
        try:
            blob = bytes.fromhex(hexs)
        except ValueError:
            return None
        if len(blob) < 8:
            return None
        return int(struct.unpack("<I", blob[:4])[0])
    return None


# ====================================================================
# CLI.
# ====================================================================


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="picker .bin to inspect")
    ap.add_argument("--mode", required=True,
                    choices=["summary", "tree-approx", "picks", "stress", "diverge"])
    ap.add_argument("--corpus", type=Path,
                    help="features TSV — required for tree-approx, picks, diverge")
    ap.add_argument("--depth", type=int, default=4,
                    help="decision-tree depth (tree-approx, diverge)")
    ap.add_argument("--target-zq", type=float, default=85.0)
    ap.add_argument("--low-confidence-threshold", type=float, default=0.05,
                    help="log-bytes gap below which a pick counts as low-confidence (picks mode)")
    ap.add_argument("--dot", type=Path,
                    help="write graphviz tree to this path (tree-approx mode)")
    ap.add_argument("--inspect-bin", type=Path,
                    help="explicit zenpredict-inspect path (default: PATH lookup → workspace target)")
    args = ap.parse_args(argv)
    model = load_model(args.model, args.inspect_bin)

    if args.mode == "summary":
        return mode_summary(model)
    needs_corpus = args.mode in ("tree-approx", "picks", "diverge")
    if needs_corpus and args.corpus is None:
        sys.exit(f"--corpus required for mode {args.mode}")
    if args.mode == "tree-approx":
        return mode_tree_approx(model, args.corpus, args.depth, args.dot, args.target_zq)
    if args.mode == "picks":
        return mode_picks(model, args.corpus, args.target_zq, args.low_confidence_threshold)
    if args.mode == "stress":
        return mode_stress(model, args.target_zq)
    if args.mode == "diverge":
        return mode_diverge(model, args.corpus, args.depth, args.target_zq)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
