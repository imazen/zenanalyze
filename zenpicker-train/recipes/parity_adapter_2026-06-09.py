#!/usr/bin/env python3
"""Adapt the 2026-04-29 zenjpeg scalar sweep into the zenpicker-train
unified-parquet schema, for the ch4 scalar hybrid-heads parity run.

The old sweep encodes its knobs in `config_name` (e.g.
`ycbcr_444_hyb145_cs100_sa`) and keeps image features in a sibling TSV.
zenpicker-train wants one parquet with `knob_tuple_json` (categorical
cells + scalar axes as fields) + `feat_*` + `score_zensim` +
`encoded_bytes` + `codec`. This bridges the two, mirroring zentrain's
`parse_config_name` (zenjpeg_picker_config.py):

  color   = ycbcr | xyb                 (categorical)
  sub     = 444 | 420 | …               (categorical)
  trellis_on = (config has hyb<N>)      (categorical)
  sa      = config ends with _sa        (categorical)
  chroma_scale = cs<N> / 100            (scalar)
  lambda  = hyb<N> / 10, else 0.0       (scalar; 0 = trellis-off sentinel)

The cell key is the four categorical fields; chroma_scale + lambda are
the scalar heads (excluded from the cell by zenpicker-train).

Usage: parity_adapter_2026-06-09.py <sweep.parquet> <features.tsv> <out.parquet>
"""

import json
import re
import sys

import pandas as pd
import pyarrow.parquet as pq

CFG_RE = re.compile(
    r"^(?P<color>ycbcr|xyb)_(?P<sub>\d+)_(?P<trel>noT|hyb\d+)_cs(?P<cs>\d+)(?P<sa>_sa)?$"
)


def parse_config_name(name: str) -> str:
    m = CFG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable config name: {name!r}")
    trel = m.group("trel")
    trellis_on = trel != "noT"
    lam = (int(trel[3:]) / 10.0) if trellis_on else 0.0
    return json.dumps(
        {
            "color": m.group("color"),
            "sub": m.group("sub"),
            "trellis_on": trellis_on,
            "sa": m.group("sa") is not None,
            "chroma_scale": int(m.group("cs")) / 100.0,
            "lambda": lam,
        },
        separators=(",", ":"),
    )


def main(sweep_path: str, tsv_path: str, out_path: str) -> None:
    sweep = pq.read_table(
        sweep_path,
        columns=["image_path", "size_class", "config_name", "q", "bytes", "zensim"],
    ).to_pandas()
    print(f"sweep rows: {len(sweep):,} | unique configs: {sweep.config_name.nunique()}")

    # Parse each unique config_name once, then map.
    uniq = {n: parse_config_name(n) for n in sweep.config_name.unique()}
    sweep["knob_tuple_json"] = sweep.config_name.map(uniq)

    feats = pd.read_csv(tsv_path, sep="\t")
    feat_cols = [c for c in feats.columns if c.startswith("feat_")]
    print(f"features: {len(feats):,} (image,size) rows | {len(feat_cols)} feat_* cols")
    feats = feats[["image_path", "size_class", *feat_cols]]

    df = sweep.merge(feats, on=["image_path", "size_class"], how="inner")
    print(f"joined rows: {len(df):,} (dropped {len(sweep) - len(df):,} unmatched)")

    df = df.rename(columns={"bytes": "encoded_bytes", "zensim": "score_zensim"})
    df["codec"] = "zenjpeg"
    df = df[
        ["image_path", "size_class", "codec", "q", "knob_tuple_json",
         "encoded_bytes", "score_zensim", *feat_cols]
    ]
    df.to_parquet(out_path, compression="zstd", index=False)
    print(f"wrote {out_path} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
