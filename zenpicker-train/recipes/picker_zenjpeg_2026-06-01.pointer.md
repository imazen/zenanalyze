# zenjpeg picker (Profile A, ZNPR v3) — data pointer + reproducibility

**Date:** 2026-06-01. Large data lives on R2 (not git, per >30 KB rule); the
scripts + this pointer + the methodology are in git. Full method:
[`METHODOLOGY_zenpredict_caps_2026-06-01.md`](./METHODOLOGY_zenpredict_caps_2026-06-01.md).

## Git (this repo, zenanalyze)

- Trainer + per-feature input shaping + `--eval-bake`: commit `f7bb5915`.
- `recipes/rebuild_sourcefeat_parquet.py` — the join-fix that produced the
  real-feature training parquet (sha256 `90e896ceb1d58891…`).
- `recipes/METHODOLOGY_zenpredict_caps_2026-06-01.md` — method + capability
  matrix + repro chain.
- Teacher: `scripts/teacher_soft_targets.py` (Python sklearn 1.7.2 HistGB;
  the only teacher — no Rust GBM exists).

## R2 (bucket `zentrain`)

Endpoint `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com`, creds
`~/.config/cloudflare/r2-credentials` (`R2_ACCESS_KEY_ID` /
`R2_SECRET_ACCESS_KEY` → `AWS_*`). Canonical prefix:
**`s3://zentrain/picker-zenjpeg-2026-06-01/`**

| R2 path | sha256 (16) | bytes | role |
|---|---|---|---|
| `train/picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet` | `9c735461213e5a27` | 2,878,673 | **canonical training input** (334,080 rows; 320 imgs × 1044; real 108 source feats + zq_norm; encoded_bytes + Profile-A score per cell) |
| `train/picker_dense_full_zenjpeg_A_sourcefeat_BROKEN_allzero.parquet` | `17057014cd00b144` | 2,723,266 | the buggy all-zero-feature input (audit only — DO NOT train on this) |
| `source/zenjpeg_source_features_full.tsv` | `bcacb8d8f040dc93` | 374,796 | 108 source features per (image,size) variant, real values |
| `source/feature_order.txt` | `dc6882fe6168447c` | 3,321 | `feat_i` → zenanalyze feature name (runtime extraction order) |
| `scripts/rebuild_sourcefeat_parquet.py` | `90e896ceb1d58891` | 4,352 | join-fix (also in git here) |
| `METHODOLOGY_zenpredict_caps_2026-06-01.md` | `1977bd6c070aecf6` | ~11 KB | method (also in git here) |
| `bakes/picker_zenjpeg_A_FIXED_none_v3.bin` | `cb858c8abe1cf2ae` | 59,571 | **shipped FIXED bake** (f32, [64,64] distilled, no shaping; held-out SROCC 0.906, overhead 3.39%) |
| `bakes/picker_zenjpeg_A_FIXED_none_v3_f16.bin` | `5b807ce292dfda13` | 29,870 | f16 quant (2.0×, safe) |
| `bakes/picker_zenjpeg_A_FIXED_none_v3_i8.bin` | `888221ce25c64fb2` | 17,001 | i8 quant (3.5×, overhead-neutral) |
| `bakes/picker_zenjpeg_A_FIXED_auto_v3.bin` | — | — | input-shaping arm (FALSIFIED: SROCC 0.84; audit only, do not ship) |

## Reproduce

```sh
set -a; . ~/.config/cloudflare/r2-credentials; set +a
export AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY
EP=https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com
aws s3 cp s3://zentrain/picker-zenjpeg-2026-06-01/train/picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet ./FIXED.parquet --endpoint-url $EP
cargo build --release -p zenpicker-train --bin zenpicker-train          # commit f7bb5915
./target/release/zenpicker-train --input ./FIXED.parquet --codec zenjpeg --out picker.bin --distill --input-shaping auto --seed 0
./target/release/zenpicker-train --input ./FIXED.parquet --codec zenjpeg --eval-bake picker.bin
zenpredict repack picker.bin picker_f16.bin --dtype f16 --zerobias 0.005 --compress
```

**Headline:** the join fix took held-out bytes-SROCC from 0.34 (the BROKEN,
dial-only parquet) to 0.93 (the FIXED, real-feature parquet). Train only on
the FIXED parquet.
