# R2 `zentrain` bucket — layout and consumption

The `zentrain` bucket on Cloudflare R2 is the durable home for picker / regression
training inputs and validation outputs. Local `benchmarks/` directories under
each codec crate are the working copies; R2 is the canonical published copy that
training scripts and downstream consumers read from.

## Endpoint

```
https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com
```

Credentials live in `~/.config/cloudflare/r2-credentials` (mode 0600).

```bash
set -a && . ~/.config/cloudflare/r2-credentials && set +a
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION=auto
E="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
aws s3 ls s3://zentrain/ --endpoint-url "$E" --region auto
```

## Layout

```
s3://zentrain/
  _manifest.json                            # uploader, sha256, rows/columns per file
  <codec>/
    pareto/<basename>.parquet               # main sweep oracle (Parquet preferred)
    pareto/<basename>.tsv                   # legacy TSV form, kept only when small
    features/<basename>.{tsv,parquet}       # per-image feature vectors
    loo/<basename>.tsv                      # leave-one-out multi-seed validation
```

Codec directories: `zenwebp`, `zenjpeg`, `zenavif`, `zenjxl`. All lowercase.

`_manifest.json` is the source of truth for what's uploaded. Each entry has
`{codec, kind, key, size_bytes, sha256}` plus `{rows, columns}` for Parquet
artifacts. Inspect with `aws s3 cp s3://zentrain/_manifest.json - | jq .`.

## Pareto oracles

These are the dense quality + size sweeps from which pickers and target-zensim
regressions are trained. Parquet is canonical (zstd-compressed, ~10x smaller
than TSV, ~36x faster to load via PyArrow).

| Codec | Key | Rows | Cols | Size |
|-------|-----|------|------|------|
| zenwebp | `zenwebp/pareto/zenwebp_pareto_2026-05-01_combined.parquet` | 21,841,920 | 12 | 187 MiB |
| zenjpeg | `zenjpeg/pareto/zq_pareto_2026-04-29.parquet` | 3,497,760 | 11 | 48 MiB |
| zenjxl  | `zenjxl/pareto/zenjxl_lossy_pareto_2026-05-01.parquet` | 610,593 | 11 | 6.2 MiB |
| zenjxl  | `zenjxl/pareto/zenjxl_lossless_pareto_2026-05-01.parquet` | 165,477 | 10 | 2.3 MiB |

zenavif Pareto sweep has not yet been re-run in this format; only feature
vectors and LOO results are present for that codec today.

## Reading from R2 in `train_hybrid.py`

`zentrain/tools/train_hybrid.py` already has Parquet support via
`_read_table_columns()` (auto-detected by `.parquet` / `.pq` suffix). It reads
local `Path` objects today. To consume R2 directly, two equivalent options:

### Option A — pull a local cache, then point train_hybrid at it

```bash
mkdir -p /mnt/v/cache/zentrain
aws s3 sync s3://zentrain/zenwebp /mnt/v/cache/zentrain/zenwebp \
    --endpoint-url "$E" --region auto

python -m zentrain.tools.train_hybrid \
    --pareto /mnt/v/cache/zentrain/zenwebp/pareto/zenwebp_pareto_2026-05-01_combined.parquet \
    --features /mnt/v/cache/zentrain/zenwebp/features/zenwebp_pareto_features_2026-05-01_combined_filled.tsv \
    ...
```

This is the recommended path for repeated training runs — local SSD reads are
~1 GB/s vs ~50 MB/s from R2 over WAN.

### Option B — read directly from R2 via fsspec (no local cache)

Requires `s3fs` and treats R2 as an S3-compatible endpoint:

```python
import os, pyarrow.parquet as pq, pyarrow.fs as pafs
fs = pafs.S3FileSystem(
    endpoint_override=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    access_key=os.environ["R2_ACCESS_KEY_ID"],
    secret_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region="auto",
)
table = pq.read_table("zentrain/zenwebp/pareto/zenwebp_pareto_2026-05-01_combined.parquet", filesystem=fs)
```

For training runs that iterate the oracle many times, prefer Option A.

## Uploading a new sweep

1. Convert TSV to Parquet (zstd) using `benchmarks/tsv_to_parquet.py` in
   `zenanalyze`. Source TSVs > 1 GB stay local — only the Parquet form is
   published.
2. Add the file to a fresh upload plan TSV with columns
   `codec, kind, local_path, target_key, size_bytes, sha256`.
3. Upload with `aws s3 cp ... --endpoint-url "$E" --region auto`.
4. Regenerate `_manifest.json` from the plan and the local file inventory and
   upload it (overwriting the old manifest).

## Hard rules

- The Parquet form is canonical; never delete the local Parquet after upload
  (R2 is a published copy, not the only copy).
- Do not upload source TSVs > 1 GB. Convert to Parquet first.
- LOO results stay TSV — they're tiny (< 30 KB each).
- Never bake R2 credentials into a script or commit them.
