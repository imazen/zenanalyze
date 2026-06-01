#!/usr/bin/env python3
"""Rebuild the zenjpeg picker parquet with REAL source features.

The 2026-06-01 build of `picker_dense_full_zenjpeg_A_sourcefeat.parquet`
joined the source-feature TSV onto the sweep rows with a broken key, so
every `feat_0..feat_107` column came out all-zero (silent empty->0.0
imputation). The picker therefore trained on `zq_norm` alone.

This rebuild re-joins the 108 real source features by the correct key
(`<source_stem>@sz<width>`), FAILS LOUDLY on any unmatched row (never
re-imputes zeros), keeps every other column (image_basename, codec, q,
knob_tuple_json, encoded_bytes, score_zensim, ...) intact, and writes a
`feature_order.txt` sidecar mapping feat_i -> the zenanalyze feature name
so the runtime extracts features in the same order.
"""
import csv
import sys
import pyarrow as pa
import pyarrow.parquet as pq

D = "/mnt/v/zen/picker-dense-full-2026-05-27"
SRC_PARQUET = f"{D}/parquet/picker_dense_full_zenjpeg_A_sourcefeat.parquet"
OUT_PARQUET = f"{D}/parquet/picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet"
TSV = f"{D}/zenjpeg_source_features_full.tsv"

# 1. Load the TSV: map <stem>@sz<width> -> [108 feature floats in header order].
with open(TSV) as f:
    r = csv.reader(f, delimiter="\t")
    hdr = next(r)
    idx = {c: i for i, c in enumerate(hdr)}
    feat_cols = [c for c in hdr if c.startswith("feat_")]
    feat_pos = [idx[c] for c in feat_cols]
    print(f"TSV has {len(feat_cols)} feature columns (cols {feat_pos[0]}..{feat_pos[-1]})")
    tsv_map = {}
    for row in r:
        stem = row[idx["source"]]
        if stem.endswith(".png"):
            stem = stem[:-4]
        width = row[idx["width"]]
        # width may be a float-string like "32.0"; normalise to int.
        w = str(int(float(width)))
        key = f"{stem}@sz{w}"
        # Per-cell empty -> 0.0 is legitimate: some features are
        # genuinely undefined for tiny (e.g. 32x32) images. This is
        # NOT the join-failure bug (which zeroed ALL features for ALL
        # rows); here only specific (image, feature) cells are blank.
        tsv_map[key] = [
            (float(row[p]) if p < len(row) and row[p].strip() != "" else 0.0)
            for p in feat_pos
        ]
print(f"TSV map: {len(tsv_map)} (source,size) variants")

n_feat = len(feat_cols)
if n_feat != 108:
    print(f"WARNING: expected 108 features, TSV has {n_feat}")

# 2. Load the existing parquet, replace feat_0..feat_{n-1} from the map.
table = pq.read_table(SRC_PARQUET)
names = table.column_names
ib = table.column("image_basename").to_pylist()

# Build new feature columns by lookup.
new_feats = [[0.0] * len(ib) for _ in range(n_feat)]
missing = set()
for i, key in enumerate(ib):
    vals = tsv_map.get(key)
    if vals is None:
        missing.add(key)
        continue
    for j in range(n_feat):
        new_feats[j][i] = vals[j]

if missing:
    print(f"FATAL: {len(missing)} parquet image_basename keys had NO TSV match — "
          f"refusing to impute zeros. Examples: {list(missing)[:5]}")
    sys.exit(1)
print(f"join OK: all {len(set(ib))} distinct image_basename matched a TSV variant")

# 3. Rebuild the table column-by-column, swapping feat_i.
cols = []
for name in names:
    if name.startswith("feat_"):
        j = int(name[len("feat_"):])
        if j < n_feat:
            cols.append(pa.array(new_feats[j], type=pa.float64()))
        else:
            cols.append(table.column(name))  # leave any extra feat cols
    else:
        cols.append(table.column(name))
out = pa.table(cols, names=names)

pq.write_table(out, OUT_PARQUET, compression="zstd", compression_level=3)
print(f"wrote {OUT_PARQUET} ({out.num_rows} rows, {out.num_columns} cols)")

# 4. Sidecar: feat_i -> real zenanalyze feature name (extraction order).
with open(f"{D}/parquet/feature_order.txt", "w") as f:
    for i, c in enumerate(feat_cols):
        f.write(f"feat_{i}\t{c}\n")
print(f"wrote feature_order.txt ({n_feat} names)")

# 5. Verify: feat_0 now nonzero + varies across images.
chk = pq.read_table(OUT_PARQUET, columns=["image_basename", "feat_0", "feat_1"]).to_pydict()
per = {}
for img, f0 in zip(chk["image_basename"], chk["feat_0"]):
    per.setdefault(img, f0)
vals = list(per.values())
print(f"VERIFY feat_0: distinct-across-images={len(set(vals))} min={min(vals):.4f} max={max(vals):.4f} "
      f"(was all-zero before)")
