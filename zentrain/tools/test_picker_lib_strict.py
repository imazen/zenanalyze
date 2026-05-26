#!/usr/bin/env python3
"""
Regression test for `_picker_lib.load_features_raw(strict=..., drop_nan_rows=...)`.

Shipped in two waves:

1. DEDUP-B (2026-05-26) — `strict: bool = True` kwarg added to preserve
   capacity_sweep's lenient KEEP_FEATURES handling without duplicating
   the loader. True (default) raises SystemExit on missing columns;
   False silently drops them.

2. DEDUP-B3 (2026-05-26) — `drop_nan_rows: bool = False` kwarg added so
   `student_permutation.load_features` can delegate to the canonical
   loader. True drops rows whose feature values contain NaN (tiny
   images skipping percentile features per zenanalyze #49); False
   (default) keeps NaN values in the row's feature vector. The flag
   also makes empty-string cells parse to NaN regardless (the flag
   only controls keep-vs-drop).

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

# 4-row features TSV mixing clean / NaN-text / empty-string rows; powers
# the drop_nan_rows test cases.
TSV_WITH_NANS = (
    "image_path\tsize_class\tfeat_a\tfeat_b\n"
    "img1.png\tsmall\t1.0\t2.0\n"
    "img2.png\tlarge\tnan\t4.0\n"
    "img3.png\ttiny\t\t6.0\n"
    "img4.png\tmedium\t7.0\t8.0\n"
)


def _write_tsv(d: Path) -> Path:
    p = d / "fx.tsv"
    p.write_text(TSV_CONTENTS)
    return p


def _write_nan_tsv(d: Path) -> Path:
    p = d / "fx_nan.tsv"
    p.write_text(TSV_WITH_NANS)
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

    # Cases 6-9 use the NaN-bearing fixture (4 rows: img1 clean, img2
    # nan-text in feat_a, img3 empty string in feat_a, img4 clean).
    with tempfile.TemporaryDirectory() as d:
        p = _write_nan_tsv(Path(d))

        # Case 6: drop_nan_rows=False (default) -> all 4 rows kept,
        # NaN values surface in the feature vectors.
        import math
        feats, cols = _picker_lib.load_features_raw(p, None)
        if len(feats) != 4:
            fails.append(f"CASE 6: expected n_rows=4, got {len(feats)}")
        else:
            v2 = feats[("img2.png", "large")]
            v3 = feats[("img3.png", "tiny")]
            if not (math.isnan(v2[0]) and math.isnan(v3[0])):
                fails.append(
                    f"CASE 6: expected NaN in img2/img3 feat_a, "
                    f"got img2={v2}, img3={v3}"
                )
            else:
                print(
                    f"CASE 6 OK: drop_nan_rows=False -> "
                    f"n_rows={len(feats)}, NaN values preserved"
                )

        # Case 7: drop_nan_rows=True -> img2 + img3 dropped, 2 rows left.
        feats, cols = _picker_lib.load_features_raw(
            p, None, drop_nan_rows=True
        )
        if len(feats) != 2:
            fails.append(f"CASE 7: expected n_rows=2, got {len(feats)}")
        elif ("img2.png", "large") in feats or ("img3.png", "tiny") in feats:
            fails.append(
                f"CASE 7: img2/img3 should have been dropped, "
                f"got keys={list(feats.keys())}"
            )
        else:
            print(
                f"CASE 7 OK: drop_nan_rows=True -> "
                f"n_rows={len(feats)} (img2/img3 dropped)"
            )

        # Case 8: drop_nan_rows=True with KEEP_FEATURES filter on a clean
        # subset (feat_b never has NaN) -> all 4 rows kept.
        feats, cols = _picker_lib.load_features_raw(
            p, ["feat_b"], drop_nan_rows=True
        )
        if len(feats) != 4:
            fails.append(
                f"CASE 8: expected n_rows=4 (no NaN in feat_b), "
                f"got {len(feats)}"
            )
        elif cols != ["feat_b"]:
            fails.append(f"CASE 8: expected ['feat_b'], got {cols}")
        else:
            print(
                f"CASE 8 OK: drop_nan_rows=True + filter clean col -> "
                f"n_rows={len(feats)}, cols={cols}"
            )

        # Case 9: drop_nan_rows=True composes with strict=False.
        feats, cols = _picker_lib.load_features_raw(
            p, ["feat_a", "feat_missing"], strict=False, drop_nan_rows=True
        )
        if cols != ["feat_a"]:
            fails.append(
                f"CASE 9: expected cols=['feat_a'], got {cols}"
            )
        elif len(feats) != 2:
            fails.append(
                f"CASE 9: expected n_rows=2 after dropping "
                f"img2/img3 NaN in feat_a, got {len(feats)}"
            )
        else:
            print(
                f"CASE 9 OK: strict=False + drop_nan_rows=True compose -> "
                f"cols={cols}, n_rows={len(feats)}"
            )

    if fails:
        print("\nFAIL:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nALL 9 CASES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
