# metapicker_v1 (criterion-8 codec-family meta-picker) — pointer + wiring state

- **Bake**: `/mnt/v/output/zensim/metapicker-2026-08-30/metapicker_v1.bin`
  (104,193 B, sha256 `4479ef9c874ebf1c…`; sibling `.toml` manifest).
  NOT committed (>30 KB rule); this pointer is the tracked reference.
- **Contract** (self-describing via bake metadata): 62 inputs =
  61 zenanalyze source features (`zenpicker_train.image_feature_names`,
  order in `zenpicker_train.input_order`) ⊕ `zq_norm`; 7 outputs =
  `bytes_log` per cell, cells = `zenpicker_train.cell_labels`
  (family×mode: zenavif_lossy, zenjpeg_lossy, zenjxl_lossless,
  zenjxl_lossy, zenpng_lossless, zenwebp_lossless, zenwebp_lossy);
  pick = masked argmin over reachable cells.
- **Training**: `zenpicker-train --mode mlp` grid winner [128,128] lr 2e-3
  on the 7 canonical picker datasets RESCORED under zensim Profile B
  (era rule), 61 source features (36 size-gated cols dropped), origin-digit
  train view. Builder:
  `zensim/scripts/canonical_corpus/build_metapicker_input_2026-08-30.py`.
- **Honest panels**: origin-validate (odd origins, 38,668 rows):
  argmin 0.7499 · overhead mean 4.47% / p50 0 / p90 14.5% ·
  bytes-SROCC 0.9869. Baseline gate: best fixed family (always-avif)
  pays 20.4% mean / 55% p90 — v1 is 4.5× better. Record:
  zensim `benchmarks/balance_campaign_2026-08-28.md` (criterion-8 section).
- **WIRING DECISION (registered, open)**: the v1 cells are FAMILY×MODE (7)
  while `zenpicker::CodecFamily` is a 6-enum with no mode axis — a direct
  `MetaPicker::new(model)` would mis-map outputs. The adapter (either a
  mode-aware routing enum, or metadata-driven cell→(family,mode) mapping in
  `MetaPicker`) is an API design decision — proposed, not jammed in. The
  zenpredict load+forward+argmin path is already proven end-to-end by
  `evaluate_picker_bake` (the panels above run through `zenpredict::Predictor`).
- Test view (origins {7,9}) remains touch-once for the ship proposal.
