# Experimental-feature maturity audit + promotion (2026-06-13)

Decides which `#[cfg(feature = "experimental")]` features in
`src/feature.rs` graduate to the default surface, and re-scopes what
the `experimental` gate means going forward. Companion to the un-gate
commit and to `feature_redundancy_clusters_2026-06-13.md` (the
selection-guidance half).

## The corrected promotion criterion

The `experimental` gate must track **one thing only: is the feature's
*structural definition* settled and computed correctly?** It must NOT
track:

- **Per-codec signal / importance.** Feature value is a conditional
  `features × knobs × zq-band × mode` matrix; there is no global
  ranking. A feature that is a redundant twin for *zenjpeg*'s picker
  is the one *zenwebp* already leans on (zenwebp's `FEAT_COLS` keeps
  `noise_floor_y_p25/p75`, `quant_survival_y_p50/p75` that barely move
  zenjpeg). Gating on one codec's importance hides a feature the other
  three codecs can't then test. "every codec needs different feature
  sets."
- **Redundancy.** ρ≥0.9 twins are a *KEEP_FEATURES selection* concern,
  handled by docs + a lint (`zentrain/tools/lint_keep_features.py`),
  never by a compile-time gate.
- **Proven usefulness.** "Net-negative for the current small-MLP
  picker" is not a definition problem. Keeping such a feature gated
  blocks the real testing that would settle it.

Numeric/scale drift stays allowed on promoted features (the crate
threshold contract covers it); the gate is for *structural*
uncertainty only.

## Why the old "has a consumer" gate is dead

The Cargo.toml gate used to read "promote once ≥1 in-tree consumer."
That no longer discriminates: the wired zenjpeg picker
(`zenjpeg/src/encode/picker.rs` → `picker_data/feature_order.txt`)
ingests `feat_0..feat_107` — **nearly the entire schema, including the
HVS pack and `spectral_slope_y` that the 2026-05-17 eval measured as a
regression.** Being in a bake ≠ being validated. So promotion is now
decided on definition-maturity (above), not consumer presence.

## Decision

**Promoted to default (58 features)** — pinned, deterministic
definitions, a month-plus of multi-codec testing:

- Tier 1: `colourfulness`, `laplacian_variance`, `variance_spread`
- Tier 3 DCT/AQ: `dct_compressibility_y/uv`, `patch_fraction`,
  `patch_fraction_fast`, `aq_map_mean/std`, `aq_map_p{1,5,10,50,75,90,95,99}`,
  `noise_floor_y/uv`, `noise_floor_y_p{1,5,10,25,50,75,90}`,
  `noise_floor_uv_p{25,50,75,90}`, `quant_survival_y/uv`,
  `quant_survival_y_p{1,5,10,25,50,75}`, `quant_survival_uv_p{10,25,50,75}`,
  `gradient_fraction`, `gradient_fraction_smooth`, `luma_kurtosis`
- Tier 1 piggyback: `grayscale_score`, `skin_tone_fraction`, `edge_slope_stdev`,
  `laplacian_variance_p{1,5,10,50,75,90,99}`, `laplacian_variance_peak`
- Palette: `palette_fits_in_256`, `palette_log2_size`
- HVS (pinned formulas — Pearson cov / Wang-Li IW-SSIM / gradient
  anisotropy / Field 1/f): `chroma_luma_covariance_cb/cr`,
  `info_weight_mean/p90`, `orientation_energy_ratio`, `spectral_slope_y`

The HVS/spectral group is promoted **despite** being net-negative for
the current pickers (`hvs_features_picker_eval_2026-05-17.md`): that is
a usefulness finding, not a definition problem, and exposing them is
what lets the "bigger-head / TabM" retest actually happen.

**Kept gated (3 features)** — genuine structural uncertainty or
retirement:

- `palette_density` (12) — `#[deprecated]` 2026-05-02, ρ=1.0 with
  `distinct_color_bins`; being retired, not promoted.
- `xyb444_color_loss` (138) — definition **rebaked within the last
  month** (binary predicate → graded Oklab ΔE); not yet settled.
- `xyb_bquarter_chroma_loss` (139) — one week old (2026-06-06), shares
  the single gated `xyb_color_loss` compute module with 138. Holds one
  cycle with its sibling. (It *is* RD-validated + consumed by
  `zenjpeg/src/encode/adaptive.rs`; revisit next cycle to promote.)

## Ablation evidence (the "last month" of runs)

Promotion rests on definition-maturity, but the signal record backs
the value of the core set. See `ablation-run-inventory` for the full
list; load-bearing inputs:

- `picker_tree_ab_importance_2026-06-09.md` — GBDT permutation
  importance (zenjpeg): `quant_survival_y` #3 individual; chroma
  cluster (`quant_survival_uv`, `noise_floor_uv`, `dct_compressibility_uv`)
  top at +0.051; luma cluster (`noise_floor_y`, `quant_survival_y`)
  +0.046.
- `feat_xform_2026-05-17_summary.md` + `seed_stable_*_2026-05-17/` —
  per-codec screen lift + seed stability for the whole family.
- `feature_groups_2026-05-02.md` + `picker_tree_ab` ρ-groups →
  `feature_redundancy_clusters_2026-06-13.{md,json}` (per-codec).

## What changed in code

- `src/feature.rs`: removed the `experimental` gate from the 58
  promoted variants + the `TIER1_*` / `TIER3_*` / `DCT_NEEDED_BY` /
  `PALETTE_QUICK_FEATURES` dispatch presets; kept it on `palette_density`,
  `xyb444_color_loss`, `xyb_bquarter_chroma_loss` and the
  `PALETTE_FULL_FEATURES` `palette_density` entry.
- `src/tier1.rs`, `src/tier3.rs`: removed all `experimental` positive
  gates and their `cfg(not(...))` fallback branches (every positive is
  now always-compiled).
- `src/lib.rs`: collapsed the `grayscale` / `laplacian` dispatch
  `cfg`/`not` pairs; un-gated the palette quick-signal writes; split
  the mixed palette block so only `palette_density` stays gated. xyb
  run-flags + write block + module stay gated.
- `src/tests.rs`: un-gated 19 promoted-feature tests (now run on the
  default build); kept the 3 `palette_density` + 5 `all(experimental,
  hdr)` tests gated.
- **Bug found + fixed by the un-gated tests:** `PALETTE_FULL_FEATURES`
  had a mixed `cfg` block bundling `palette_density` (keep) with
  `grayscale_score` + `palette_log2_size` (promote). Leaving it gated
  dropped the two promoted features from the default *dispatch* set, so
  requesting them on a default build never triggered the full palette
  scan → silent zero. Split so the two promoted features are always in
  the set.

## Validation

- `cargo test -p zenanalyze` (default): **134 passed / 0 failed**
  (incl. the 19 newly-un-gated feature tests — the silent-zero guard).
- `cargo test -p zenanalyze --features experimental`: **162 passed / 0
  failed**.
- `cargo check` clean (0 warnings) on both feature configs.
- `cargo semver-checks`: see CHANGELOG / commit (additive — gated
  variants becoming default-available is additive on a
  `#[non_exhaustive]` enum).
