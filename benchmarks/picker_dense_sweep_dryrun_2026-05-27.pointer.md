# Pointer — picker dense-q sweep dry-run data (2026-05-27)

Data for `benchmarks/picker_dense_sweep_dryrun_2026-05-27.md`. Large
files live on block storage, NOT in git (CLAUDE.md >30 KB rule).

**Block-storage root:** `/mnt/v/zen/picker-dense-dryrun-2026-05-27/`
(95 MB total)

| file | sha256 | notes |
|---|---|---|
| `parquet/picker_dense_dryrun_zenjpeg.parquet` | `c29703e88813c2686e9a8d0581c8db74821b1f1bfd42a75824e8556a0cbb32bf` | 1,856 rows × 382 cols; picker training input |
| `bake/zenjpeg_picker_dense_dryrun.bin` | `caf8ece20dc74e5b8c21140a1293017f5b0725be67cb36cf651534c67fa61a09` | ZNPR v3, 373→128→128→4, 270 KB |
| `artifacts/<sha256>.jpg` | (each filename = its sha256) | 896 content-addressed encoded variants, ~27 MB |
| `features/feat_{sz512,sz256}.parquet` | — | 372-feat zensim sidecars, 928 rows each |
| `pareto_{sz512,sz256}.tsv`, `score_pairs_ssim2.tsv` | — | per-cell scores incl ALL metric variants |
| `bake/...bin.toml` | — | picker manifest (q_is_input=false, cells, zq grid) |
| `chosen_sources.txt`, `cluster_assignment.json` | — | k-means K=8 source selection provenance |

**Provenance:**
- Source pool: `/mnt/v/input/zensim/sources/` (17,083 PNGs; 512sq variants)
- zenmetrics commit (sweep tooling): `aa3be124`
- zenanalyze commit (picker + scripts): see this commit
- zen-metrics built with `--features sweep,gpu,gpu-cuda`; local RTX 5070 CUDA
- score_zensim = batch ssim2-gpu (reach-ladder; zensim metric has a
  photo-content non-monotonicity defect — see methodology doc)

**Not mirrored to R2/Tower** — dry-run scratch, regenerable from the
scripts. The FULL run's canonical parquet + artifacts MUST be mirrored
to R2 + Tower before any cleanup (CLAUDE.md ML data discipline).

**Regenerate:** `zenmetrics/scripts/sweep/picker_dense_sweep_dryrun.sh`
→ `zenanalyze/zenpicker-train/scripts/build_picker_parquet.py`
→ `zenpicker-train --codec zenjpeg --input ... --out ...`.
