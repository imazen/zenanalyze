#!/usr/bin/env python3
"""Rebuild the zenjpeg picker training parquet with only the TOP-K source
features, renumbered feat_0..feat_{K-1} in CV-importance order.

The Rust `build_picker_dataset` loader collects every `feat_*` column,
sorts them by their numeric index, and feeds that order (then `zq_norm`)
to the MLP. So to ship a top-K picker we KEEP only the K most-important
source columns and RENUMBER them feat_0..feat_{K-1} in importance order;
the loader's numeric sort then preserves importance order, and the bake's
`feature_names` (= the new feat_i column names) line up positionally with
how `evaluate_picker_bake` feeds raw rows. We drop SOURCE-FEATURE columns
only — image_basename / codec / q / knob_tuple_json / encoded_bytes /
score_zensim (the cell structure + targets) stay byte-identical.

Inputs:
  --in-parquet   the FIXED 108-feature training parquet (real features).
  --ranking-npz  cv_ranking.npz from cv_topk.py (agg_ranked_feat_col /
                 agg_ranked_feat_name in importance order).
  --in-feature-order  the 108-row feat_i -> realname map of --in-parquet.
  --k            how many top features to keep.
Outputs:
  --out-parquet  top-K parquet (feat_0..feat_{K-1} + structural cols).
  --out-feature-order  K-row feat_i -> realname map (importance order) —
                 the order the bake consumes; ship this with the bake.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STRUCT_COLS = [
    "image_basename",
    "codec",
    "q",
    "knob_tuple_json",
    "encoded_bytes",
    "score_zensim",
]


def load_in_feature_order(path: Path) -> dict[str, str]:
    """feat_i -> realname for the input parquet's columns."""
    m = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        i, n = line.split("\t")
        m[i] = n
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", type=Path, required=True)
    ap.add_argument("--ranking-npz", type=Path, required=True)
    ap.add_argument("--in-feature-order", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-parquet", type=Path, required=True)
    ap.add_argument("--out-feature-order", type=Path, required=True)
    args = ap.parse_args()

    in_fo = load_in_feature_order(args.in_feature_order)

    r = np.load(args.ranking_npz, allow_pickle=True)
    # The aggregated ranking, importance order (best first).
    ranked_col = [str(x) for x in r["agg_ranked_feat_col"]]   # feat_0..feat_107
    ranked_name = [str(x) for x in r["agg_ranked_feat_name"]]  # real zenanalyze names

    k = args.k
    keep_cols = ranked_col[:k]      # input-parquet column names to keep
    keep_names = ranked_name[:k]    # their real zenanalyze names

    # Cross-check the ranking's name map against the input parquet's
    # feature_order.txt — both must agree on feat_i -> realname.
    for col, name in zip(keep_cols, keep_names):
        fo_name = in_fo.get(col)
        if fo_name is None:
            sys.exit(f"FATAL: ranking col {col} not in input feature_order.txt")
        if fo_name != name:
            sys.exit(
                f"FATAL: name mismatch for {col}: ranking says {name!r}, "
                f"feature_order.txt says {fo_name!r}"
            )

    table = pq.read_table(args.in_parquet)
    have = set(table.column_names)
    for sc in STRUCT_COLS:
        if sc not in have:
            sys.exit(f"FATAL: input parquet missing structural column {sc!r}")
    for col in keep_cols:
        if col not in have:
            sys.exit(f"FATAL: input parquet missing feature column {col!r}")

    # Build the output table: structural cols verbatim, then the K kept
    # feature columns RENUMBERED feat_0..feat_{K-1} in importance order.
    out_cols = []
    out_names = []
    for sc in STRUCT_COLS:
        out_cols.append(table.column(sc))
        out_names.append(sc)
    for new_i, src_col in enumerate(keep_cols):
        out_cols.append(table.column(src_col))
        out_names.append(f"feat_{new_i}")

    out = pa.table(out_cols, names=out_names)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out_parquet, compression="zstd", compression_level=3)
    sys.stderr.write(
        f"[topk-parquet] wrote {args.out_parquet} "
        f"({out.num_rows} rows, {out.num_columns} cols; "
        f"{len(STRUCT_COLS)} struct + {k} feat)\n"
    )

    # Emit the top-K feature_order.txt: feat_i -> realname (importance order).
    args.out_feature_order.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_feature_order, "w") as fh:
        for new_i, name in enumerate(keep_names):
            fh.write(f"feat_{new_i}\t{name}\n")
    sys.stderr.write(
        f"[topk-parquet] wrote {args.out_feature_order} ({k} names, importance order)\n"
    )

    # Sanity: confirm the kept feature columns are non-constant across images.
    chk = pq.read_table(
        args.out_parquet, columns=["image_basename", "feat_0", f"feat_{k-1}"]
    ).to_pydict()
    per0, perlast = {}, {}
    for img, f0, fl in zip(chk["image_basename"], chk["feat_0"], chk[f"feat_{k-1}"]):
        per0.setdefault(img, f0)
        perlast.setdefault(img, fl)
    sys.stderr.write(
        f"[topk-parquet] VERIFY feat_0 ({keep_names[0]}): "
        f"distinct-across-images={len(set(per0.values()))}; "
        f"feat_{k-1} ({keep_names[-1]}): distinct={len(set(perlast.values()))}\n"
    )

    # Report whether any hdr feature survived (so the integrator knows
    # whether the zenanalyze/hdr dependency can be dropped).
    hdr_kept = [n for n in keep_names if "hdr" in n or "luminance_nits" in n
                or "gamut" in n or "wide_gamut" in n or "effective_bit_depth" in n]
    print("HDR_FEATURES_KEPT=" + ("NONE" if not hdr_kept else ",".join(hdr_kept)))
    print("N_KEPT=" + str(k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
