# imazen-26 feature datasets — qualified-column regeneration (2026-06-23) — data pointer

> **SUPERSEDED LATER THE SAME DAY (re-bless `1c7ae48f`).** The HDR f32-kernel work
> added a second golden config pass (gamma + linear-light), which bumped **every**
> feature hash. The canonical SDR artifacts are now the **`*_requalified.parquet`**
> versions, re-headered to the final `name@hex8` names (e.g. `variance@8e9a50f1`):
> `imazen26_features_2026-06-23_requalified.parquet` (246,819 rows) and
> `imazen26_train_features_2026-06-23_requalified.parquet` (1,482 rows). **Values are
> byte-identical** to the names-only versions below (re-header, not re-extract — the
> default gamma path is unchanged by the linear-light work; verified
> `variance` = 2736.566406 either way). HDR features were re-extracted under
> linear-light instead — see
> [`imazen26_hdr_grid_2026-06-23.pointer.md`](imazen26_hdr_grid_2026-06-23.pointer.md).

Regeneration of the imazen-26 feature datasets so every feature column carries its
**qualified `name@hex8` zenanalyze-api contract identity** instead of the bare
`feat_<name>` form. **Feature values are byte-identical** to the 2026-06-22
artifacts — only the column *names* changed (the per-feature code version is now
folded into the name). Supersedes
[`imazen26_features_2026-06-22.pointer.md`](imazen26_features_2026-06-22.pointer.md)
as the canonical full-feature artifact. Large data lives in block storage + Tower,
NOT git.

**Encodes and metrics were NOT touched.** Feature parquets are separate files from
the content-hash-keyed encode artifacts and metric-score sidecars; regenerating
features reads neither, so every existing encode/metric remains valid. The qualified
schema is what the zentrain loaders (and any future bake) negotiate against — they
accept both forms, so this is a non-breaking, gradual migration.

## Artifacts (block storage `/mnt/v/output/imazen-26-features/`, Tower-mirrored ✓)

| dataset | rows × cols | sha256 | size |
|---|---|---|---|
| `imazen26_features_2026-06-23.parquet` | 246,819 × 119 (9 meta + 110 `name@hex8`) | `b113449d35e98bda0f3266fc7a25ee5ab24b92a3991cb7140e9262d4320735bc` | 60,984,763 |
| `imazen26_train_features_2026-06-23.parquet` | 1,482 × 114 | `a52348e3d07e52e70a48755e5ffa22083f9284e2963f955c4740c71fb7a6b960` | 662,681 |
| `imazen26_hdr_features_2026-06-23.parquet` | 76 × 116 | `90b2d34b3a9afd87a9d9da41284c4169677e69d35d35fbc504e58f793f09085d` | 85,093 |
| `imazen26_hdr_grid_features_2026-06-23.parquet` | 1,216 × 116 | `5ed9007e800986f88aa76735366a8aafde1cc9e5704300b9c4d68ae54a95ecd3` | 646,751 |

Tower mirror: `/mnt/tower/output/imazen-26-features/<name>_2026-06-23.parquet` (all 4
sha256-verified local == Tower). Raw full TSV kept alongside
(`imazen26_features_2026-06-23.tsv`, 280 MB).

## Provenance

- **build_commit:** `5601bf9d` (`feat(zenanalyze): imazen-26 extractors emit qualified
  name@hex8 headers under --features api`) — the extractors now map each
  `FeatureSet::SUPPORTED` column to its `versioning::feature_qualified_names()`
  identity. The qualified names are the golden-blessed
  `benchmarks/feature_qualified_names.tsv` (110), kept in sync by the
  `feature_qualified_names_match_committed` tripwire.
- **Full SDR set:** re-extracted natively via
  `examples/extract_features_imazen26_crops.rs --features experimental,hdr,api`
  (same 2157-image manifest, 11 crops, downscale-only size grid as 2026-06-13/22).
  24 threads, **0 decode failures**, 13:43 wall, peak RSS 4.77 GiB.
- **Train + HDR (native + grid):** qualified via `benchmarks/qualify_parquet_columns.py`
  (column rename `feat_<name>` → `<name>@hex8` from the committed table; values + KV
  metadata untouched) — the side-effect-free "edit parquets" path, avoiding an image
  re-render. Cross-checked: the full re-extraction and the renames produce the
  *identical* 110 qualified column names (e.g. `variance@df1b076e`).

## Validation

`benchmarks/validate_reextract.py` (now canonical-aware — matches `feat_X` ↔ `X@hex8`
by canonical name) on the full set, 2026-06-22 (bare) vs 2026-06-23 (qualified):

```
old: 246819 rows x 119 cols
new: 246819 rows x 119 cols
0 / 110 features changed:
OK — 0 feature(s) changed; the other 110 are byte-identical across the rename.
```

Byte-identical because no feature *computation* changed between the 2026-06-22 build
(`6cb86df9`) and `5601bf9d` — the intervening commits are the zenanalyze-api contract
(additive), the qualified-name namespace (column names only), and the
zenanalyze-api 0.1.0 release (docs).

## Regenerate

```bash
# Full set (native re-extraction):
cargo build --release --features experimental,hdr,api --example extract_features_imazen26_crops
target/release/examples/extract_features_imazen26_crops \
  --manifest /mnt/v/output/imazen-26-features/imazen26_manifest.tsv \
  --output   /mnt/v/output/imazen-26-features/imazen26_features_<DATE>.tsv
python3 benchmarks/tsv_to_parquet.py --keep-tsv <…>.tsv
python3 benchmarks/validate_reextract.py <2026-06-22>.parquet <new>.parquet   # expect 0 changes

# Train / HDR (rename existing parquet columns to qualified — values untouched):
python3 benchmarks/qualify_parquet_columns.py <bare>.parquet <qualified>.parquet
```
