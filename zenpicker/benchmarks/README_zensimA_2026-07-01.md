# zenpicker routers — retrained on zensim-A (2026-07-01)

The 3 baked `zenpicker_router_{lossy,lossless,gate}_v0.1.bin` in this dir were **retrained
on the profile-A canonical** (`canonical-picker-2026-07-01-zensimA`, `score_zensim` =
zensim-A / v47-strict-QAT), superseding the V0_2-scored bakes, per the "ban V0_2 labels"
directive. Same file paths / architectures / `MetaPicker::default_routers()` wiring — only
the weights (and the lossy fixture margins in `../src/lib.rs`) changed.

## Held-out TEST (7/9 origins) family accuracy on A labels

| router | arch | A held-out TEST | old V0_2 | i8 vs f32 |
|---|---|---|---|---|
| **lossy** | 6 pairwise linear discriminants + round-robin (f32) | round-robin RD overhead **mean 7.16% / p90 22.05%** | 3.55% / 12.41% | f32 (i8 flips near-0 jxl:avif) |
| **lossless** | MLP 101→128→64→6 (i8) | **89.73%** family-acc | 88.4% | i8 == f32 (Δ 0) |
| **gate** | MLP 102→128→64→2 (i8) | **98.18%** family-acc | 98.1% | i8 == f32 (Δ +0.02) |

- **lossless + gate MATCH/beat V0_2** on A.
- **lossy RD overhead is higher under A** (7.16% vs V0_2's 3.55%) — a genuine A-vs-V0_2
  shift: the metric ranks lossy codecs differently (cf. metric-codec-bias-ssim2-vs-zensim).
  Same pairwise architecture, retrained; NOT a regression in the picker, a property of A.

## Provenance / how to reproduce

- **Labels**: unbiased support-aware oracle via `picker_data.oracle_rows(require='all')`
  (lossy targets 45–90 step 3; pairwise 48–88 step 4) — only cells where ALL required
  families have MEASURED support (no extrapolation bias). Cross-codec quality-coverage gate
  FAILS above zq90 on A (webp/jxl under-cover, avif reaches 100%) — the require='all' layer
  excludes those biased high-band cells by construction (28536 support-complete train cells,
  8376 excluded, mostly jxl/webp high-band gaps).
- **Lossless identification on A**: profile A saturates at **97.6893** and NEVER reaches 100
  even on byte-identical reconstructions (V0_2 scored the same encodes at 100.0). So the old
  `score>=99.999` true-lossless filter yields ZERO rows on A. Lossless-ness is instead read
  from **cell semantics** (metric-free, A-consistent): a cell is lossless iff it carries no
  palette/quality-limit token (png: no `-iq`/`-zq`; webp: `vp8l` w/o `-ql<100`; jxl: all
  modular). Validated **100.00% precision** vs V0_2 ground truth; lossless oracle on A =
  jxl 87.8% / webp 12.2% / png 0% (matches the documented distribution).
- **Features**: source-only sidecar `router-features-2026-06-30/zenanalyze_features.parquet`
  (101 qualified `name@hex8` feats, experimental-complete; metric-independent, reused as-is).
- **Trainers**: zenmetrics `scripts/picker/{bake_routers,train_router_clean}.py` (MLP
  lossless/gate) + `bake_lossy_pairwise` (the pairwise lossy discriminants), all pointed at
  the A canonical. i8 repack via `zenpredict repack --dtype i8`; verified on the real `.bin`
  with `examples/score_router.rs`. Self-verify: ZNPR forward argmin == sklearn (match 1.0000);
  pairwise ZNPR margins == sklearn (max|err| 8.9e-15).
- **Verified**: `cargo test -p zenpicker --features api` — `default_routers_load_and_route`
  + `lossy_router_roundtrip_matches_fixture_margins` (fixture updated to A margins) + full
  suite (29 lib + 8 integration) green.

Block-storage originals + logs: `/mnt/v/output/pickerA-2026-07-01/routers/`.
