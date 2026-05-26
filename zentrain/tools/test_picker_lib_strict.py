#!/usr/bin/env python3
"""
Regression test for `_picker_lib.load_features_raw(strict=...)`.

Shipped as part of the Tier-1 #6 dedup migration (4 zentrain ablation
tools — feature_ablation, feature_group_ablation, validate_schema,
capacity_sweep — replaced their local `load_pareto` / `load_features`
copies with delegations to `_picker_lib.load_pareto_raw` /
`load_features_raw`).

`capacity_sweep`'s pre-dedup local `load_features` was lenient about
KEEP_FEATURES: any column absent from the TSV was silently dropped.
`_picker_lib.load_features_raw` historically raised `SystemExit` on
missing columns. To preserve capacity_sweep's behavior without
duplicating the loader, this chunk added a `strict: bool = True` kwarg
to `load_features_raw`: True (default) keeps the existing
SystemExit-on-missing semantics for `load_or_build_dataset` and the
3 strict-mode tools; False powers capacity_sweep.

Run:
    python3 zentrain/tools/test_picker_lib_strict.py

Exits 0 on success, non-zero with a diagnostic on failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add this directory to sys.path so we can `import _picker_lib` directly
# (the zentrain/tools dir is not a package; see feature_ablation.py
# train_and_eval_warm_start for the same pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _picker_lib  # noqa: E402

# 2-row features TSV with cols feat_a, feat_b.
TSV_CONTENTS = (
    "image_path\tsize_class\tfeat_a\tfeat_b\n"
    "img1.png\tsmall\t1.0\t2.0\n"
    "img2.png\tlarge\t3.0\t4.0\n"
)


def _write_tsv(d: Path) -> Path:
    p = d / "fx.tsv"
    p.write_text(TSV_CONTENTS)
    return p


def main() -> int:
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as d:
        p = _write_tsv(Path(d))

        # Case 1: no filter — all feat_ cols included.
        feats, cols = _picker_lib.load_features_raw(p, None)
        if cols != ["feat_a", "feat_b"]:
            fails.append(f"CASE 1: expected ['feat_a','feat_b'], got {cols}")
        elif len(feats) != 2:
            fails.append(f"CASE 1: expected n_rows=2, got {len(feats)}")
        else:
            print(f"CASE 1 OK: no-filter -> cols={cols}, n_rows={len(feats)}")

        # Case 2: filter to subset.
        feats, cols = _picker_lib.load_features_raw(p, ["feat_b"])
        if cols != ["feat_b"]:
            fails.append(f"CASE 2: expected ['feat_b'], got {cols}")
        else:
            print(f"CASE 2 OK: filter [feat_b] -> cols={cols}")

        # Case 3: strict=True (default) on missing col -> SystemExit.
        try:
            _picker_lib.load_features_raw(p, ["feat_missing"])
        except SystemExit as e:
            print(f"CASE 3 OK: strict=True -> SystemExit({e})")
        else:
            fails.append("CASE 3: expected SystemExit, got normal return")

        # Case 4: strict=False on missing col -> lenient (silently drop).
        feats, cols = _picker_lib.load_features_raw(
            p, ["feat_a", "feat_missing"], strict=False
        )
        if cols != ["feat_a"]:
            fails.append(f"CASE 4: expected ['feat_a'], got {cols}")
        else:
            print(f"CASE 4 OK: strict=False -> cols={cols} (missing dropped)")

        # Case 5: strict=False, all missing -> empty cols, no error.
        feats, cols = _picker_lib.load_features_raw(
            p, ["feat_x", "feat_y"], strict=False
        )
        if cols != []:
            fails.append(f"CASE 5: expected [], got {cols}")
        else:
            print(f"CASE 5 OK: strict=False all missing -> cols={cols}")

    if fails:
        print("\nFAIL:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nALL 5 CASES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
