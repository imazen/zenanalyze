# Picker training principles + cross-codec defaults

The "what's invariant across codecs" reference. Read this once before
adopting / re-baking a picker for any codec. The step-by-step is in
[`FOR_NEW_CODECS.md`](FOR_NEW_CODECS.md); the codec-side rescue
protocol is in [`SAFETY_PLANE.md`](SAFETY_PLANE.md). This doc is the
"why and what" — defaults, gates, decisions, the shape of the work.

---

## Crate map

| | What it is | Lives in |
|---|---|---|
| **zenanalyze** | Feature extractor — one pass over a `zenpixels::PixelSlice`, returns the numeric feature vector every picker / scorer consumes. | [`/`](../) |
| **zenpredict** | Generic Rust runtime. Loads a ZNPR v2 `.bin`, runs forward pass, exposes masked argmin (with scorer-composed scores), typed metadata, OOD bounds, two-shot rescue. | [`/zenpredict/`](../zenpredict/) |
| **zenpicker** | Codec-family meta-picker. Given features + target quality + allowed-family mask, picks `{jpeg,webp,jxl,avif,png,gif}`. Wraps `zenpredict::Predictor`. | [`/zenpicker/`](../zenpicker/) |
| **zentrain** | Python training pipeline. Pareto sweep harness, teacher fit, distill, ablation, holdout probes, safety reports, `.bin` bake (this directory). | [`/zentrain/`](.) |
| **Per-codec picker** | Lives in each codec crate (zenjpeg, zenwebp, zenavif, zenjxl). Depends on `zenpredict`. Trained via `zentrain` with a codec-config module. | each codec repo |
| **zensim V0_4** | Perceptual scorer. Same ZNPR v2 format, but `n_outputs = 1` and the output is a perceptual distance, not log-bytes. | `zen/zensim/` |

The contract between training and runtime is **ZNPR v2**. The
contract between feature extraction and runtime is **the codec's
`feat_cols` list + `schema_hash`**. Both are versioned independently.

---

## What a picker is

A function:

```text
(image features, target perceptual quality, caller constraints, optional cost knobs)
                                  │
                                  ▼
                  (chosen encoder config, predicted bytes, confidence)
```

The `.bin` is a small MLP (~30–200 KB) that predicts log-bytes per
config per image. Argmin under the caller's allowed-cell mask picks
the config. Everything else is composition on top of that.

**Two layers** — most pipelines stack them:

1. **Meta-picker (`zenpicker`)** — picks the codec family. One bake.
2. **Per-codec picker** — within the chosen family, picks the config.

Pure-codec pipelines (only-jpeg image proxy) skip the meta-picker.
Multi-codec pipelines (image-cdn, codec-shootout) use both.

---

## Data discipline (cross-codec, non-negotiable)

These rules apply to every picker / scorer bake. Numbers are tuned
against the global sweep discipline in
`~/work/claudehints/CLAUDE.md`; bumping them tighter is fine, looser
is not.

### 1. Sweep four dimensions every time

The corpus that trains the picker MUST cover:

- **Size**: `tiny (≤64) + small (256) + medium (1024) + large (4096)` per source image (not "different sources at different sizes" — every size for every image so the slope and intercept can both be fit).
- **Quality**: q step 5 from 0..70, q step 2 from 70..100 (≈30 points). Denser at low-q than high-q is fine; the inverse is forbidden — **low-q is where structural problems hide**.
- **Mode** (per-codec): every encoder knob the bake will eventually predict (XYB on/off, trellis on/off, every effort level, etc.). Picker can't pick what wasn't measured.
- **Content**: ≥50 photographic + ≥50 screen-content + ≥50 line-art / mixed images, named corpora (CID22, CLIC, gb82-sc, screenshots). Sealed validation hold-out ≥20 % per class.

**Random-sampling sources for "representative images" is forbidden** for trained pickers. Use k-means on the feature-space embedding to pick representative images (#47); random-sampling under-represents outliers, and outliers are where pickers fail in production.

### 2. Per-(image, size) zensim ceiling is a first-class column

(#51, PR #52.) Tiny images can't reach `zq ≥ 90` regardless of
encoder. Training the picker on `(tiny, zq=98)` cells produces
DATA_STARVED_SIZE noise. Every pareto sweep TSV MUST emit
`effective_max_zensim` per `(image, size_class)` and the trainer
filters cells where `target_zq > effective_max_zensim` before fit.

The trainer fires `UNCAPPED_ZQ_GRID` when `max(ZQ_TARGETS) > 85` and
the column is missing. Treat that violation as blocking.

### 3. Sample-count floors → NaN → result-map dropout

(#49 — landed.) Percentile features (`aq_map_p*`,
`noise_floor_y/uv_p*`, `quant_survival_y/uv_p*`,
`laplacian_variance_p*`) are statistically meaningless below
per-feature minimum sample counts (100 blocks for tier3 percentiles,
1024 interior pixels for laplacian). Below the floor, zenanalyze
emits `f32::NAN` → `AnalysisResults::set` skips → `result.get(...)`
returns `None` → codec routes via OOD-fallback path.

**Codecs MUST handle the `None` case** via either
`first_out_of_distribution` + `RescueStrategy::KnownGoodFallback` or
explicit `unwrap_or(default)` with a known-good substitute. Don't
silently propagate `0.0`.

### 4. Match training-time `pixel_budget` to runtime `pixel_budget`

(#48.) `feat_patch_fraction_fast` is the only feature that doesn't
saturate vs `pixel_budget`. Drift is `+0.054σ` at 1 MP, `+0.107σ` at
4 MP. **A picker baked at one budget MUST be re-baked when the
runtime default budget changes.** PR #44 (raise default to 2 MP) is
gated on this.

### 5. OOD bounds in every bake

The bake MUST emit `feature_bounds` (the top-level Section in ZNPR
v2). p01 / p99 per feature column from the training corpus. Codecs
read bounds at load and call `first_out_of_distribution` on every
encode.

### 6. Re-bake triggers

| Trigger | Action |
|---|---|
| zenanalyze feature mean drift > 0.10·σ on any input feature | Re-bake required |
| `schema_hash` mismatch (feat_cols changed) | Hard fail at load — re-bake required |
| Default `pixel_budget` change | Re-bake required (per #48) |
| ZNPR format version change | Re-bake required (loader rejects old version) |
| New corpus class added (e.g. mobile screenshots) | Re-bake recommended; verify no regression on existing classes |
| Encoder version bump that changes byte cost > 1 % | Re-bake required (the picker learned the old encoder's pareto front) |

---

## Argmin objective (composing scores at the codec edge)

The picker's output is a vector of predictions. The **score** the
codec argmins over is the codec's choice. zenpredict provides:

- `argmin_masked(features, mask, transform, offsets)` — log-bytes +
  uniform/per-output additive offsets. The default for "minimize
  bytes given target_zq."
- `argmin_masked_with_scorer(features, mask, |out, i| { ... })` (#55,
  landed) — caller-composed score from any combination of hybrid heads.

### Score recipes

| Goal | Recipe |
|---|---|
| **Default** — smallest bytes given `target_zq` | `ScoreTransform::Exp + ArgminOffsets { uniform: caller_icc_size, per_output: format_overhead }` |
| **Hard time cap** — must finish in ≤ N ms | Read `time_pred[i]` from hybrid heads, mask out `time_pred[i] > N` upstream, then default argmin |
| **Soft RD-vs-time** — μ bytes per ms | `\|out, i\| { out[i].exp() + mu * out[N_CELLS + i] }` via scorer |
| **Time-to-percent-saved** — pick the fastest config that saves ≥ X % bytes vs baseline | Mask cells where `predicted_bytes / baseline_bytes > 1 - X/100`, then argmin on `time_pred` |
| **Multi-metric** — caller picks `zensim / ssim2 / butter` at request time | Ship one bake per metric (see §"Metric choice" below); load the appropriate `.bin` |

### Metric choice — when to ship multiple bakes

(#56.) Three architecturally distinct options:

**A. One bake per metric** (recommended).
Ship `picker_zensim.bin`, `picker_butter.bin`, `picker_ssim2.bin`
side by side. Codec selects which to load based on caller intent.
Each bake is fully calibrated; cross-bake bias is impossible by
construction.

- **Bake cost**: train each separately (~3× corpus encode time if
  metrics share encode passes; ~9× if not).
- **Storage cost**: 3 × ~150 KB = trivial.
- **Runtime cost**: zero — pick `.bin` at startup.
- **Calibration**: each metric's pareto front is fully captured.

**B. Multi-output single bake**. One model emits `[bytes_log[N], bytes_log[N], bytes_log[N]]` indexed by metric. Argmin runs over the right slice.

- **Pros**: one .bin to ship and validate.
- **Cons**: model must learn correlations between metrics; lower
  capacity at the high-quality tail where metrics disagree most.
- **When**: metrics correlate strongly across the corpus (rule of
  thumb: ≥ 0.95 Spearman on `(image, config) → metric` rank within
  every `(size, q)` band).

**C. Metric as input feature**. One-hot for "which metric." Picker conditions.

- **Pros**: smallest model, single .bin.
- **Cons**: trusts generalization across metrics. Riskiest at the
  out-of-corpus tail. Avoid for SLA-bound deployments.

**Default recommendation: ship A.** Metrics agree on the easy 80 %
and disagree exactly on the inputs operators care about (HDR,
high-contrast UI, line-art). Calibrate each separately so the
disagreement is observable and contractually advertised, not
implicit in a single model.

### RD-vs-time bake (#56, scoped)

When the codec wants to expose a μ-style "bytes-per-ms" tradeoff:
trainer emits `time_log[N_CELLS]` as a hybrid head alongside
`bytes_log[N_CELLS]`. Codec uses
`predictor.argmin_masked_with_scorer_in_range` with a closure that
reads both heads.

Platform calibration via `zentrain.median_cell_ms_per_mp` baked in
metadata + a startup-measured per-CPU multiplier; see #56 for the
calibration protocol.

---

## Runtime contract (codec-side patterns)

Same shape every codec follows. Pseudocode:

```rust
// 1. Load + schema-hash gate.
const MODEL: &[u8] = &Aligned(*include_bytes!("…")).0;   // align(16)
let model = zenpredict::Model::from_bytes_with_schema(MODEL, MY_SCHEMA_HASH)?;

// 2. Read OOD bounds + reach gate at startup.
let bounds = model.feature_bounds();   // empty slice if absent
let reach_rates = model.metadata().get_numeric(keys::REACH_RATES).ok();

// 3. Per-encode:
let features = my_codec::extract_features(&analysis, target_zq);

// 3a. OOD detection.
if let Some(idx) = zenpredict::first_out_of_distribution(&features, bounds) {
    return RescueStrategy::KnownGoodFallback.config_for(target_zq);
}

// 3b. Build constraint mask. AND with reach gate if profile is strict.
let mut mask = caller_constraints.allowed_mask();
if profile == Profile::Strict {
    let mut gate = [false; N_CELLS];
    zenpredict::threshold_mask(&reach_rates_for(target_zq), 0.99, &mut gate);
    for (i, allowed) in mask.iter_mut().enumerate() { *allowed &= gate[i]; }
}

// 3c. Pick.
let pick = predictor.argmin_masked_in_range(
    &features, (0, N_CELLS), &AllowedMask::new(&mask),
    ScoreTransform::Exp,
    Some(&ArgminOffsets { uniform: icc_bytes, per_output: Some(&FORMAT_OVERHEAD) }),
)?.expect("at least one cell allowed");

// 3d. Two-shot rescue if the pick verifies low.
let bytes_0 = encode(pick); let achieved = verify(&bytes_0);
match should_rescue(achieved, target_zq, &policy) {
    RescueDecision::Ship   => bytes_0,
    RescueDecision::Rescue => ship_best_of(bytes_0, encode_rescue_path()),
}
```

`Predictor::argmin_masked_with_scorer` replaces step 3c when the
codec composes its own score (RD-vs-time, multi-metric).

---

## Per-codec adoption notes

Each codec writes a small `<codec>_picker_config.py` declaring its
TSV paths, KEEP_FEATURES, ZQ_TARGETS, `parse_config_name`,
`CATEGORICAL_AXES`, `SCALAR_AXES`. Reference:
[`zentrain/examples/zenjpeg_picker_config.py`](examples/zenjpeg_picker_config.py).

### zenjpeg

- **Quality scale**: zensim points
- **Categorical axes**: `(color_mode, sub, scan, sa_piecewise, trellis_on)` — typically 12 cells
- **Scalar heads**: `chroma_scale`, `lambda` (only meaningful when `trellis_on`)
- **Rescue strategy default**: `ConservativeBump` (jpegli-q + 2)
- **Bake**: `zenjpeg_picker_v2.2_full.bin`, ~30 KB f16

### zenwebp

- **Quality scale**: zensim points
- **Categorical axes**: `method` (0-6), optional `segments` (1-4) — typically 14-20 cells. Note the spike branch (`spike/zenpicker-knobs`) explored adding more categorical axes; production layout TBD until the spike lands.
- **Scalar heads**: `effort` (0-9), `sns_strength`, `filter_strength`, `filter_sharpness`
- **Rescue strategy default**: `ConservativeBump` (effort + 1, sns_strength + 10)

### zenavif (rav1d-safe + zenrav1e)

- **Quality scale**: zensim points (codec maps to AV1 cq internally)
- **Categorical axes**: `tune` (psnr/ssim/butter/perceptual) — 4 cells
- **Scalar heads**: `speed` (0-10), `cq`, optional `tile_cols`/`tile_rows`
- **Rescue strategy default**: `ConservativeBump` (cq -= 4, tune ⇒ butter)

### zenjxl (jxl-encoder)

Two distinct pickers — lossy and lossless are different shapes:

**Lossy** (VarDCT)
- **Quality scale**: distance (jxl native), mapped to/from zensim at codec edge
- **Categorical axes**: `modular_mode` (off/yes), color space (xyb/none)
- **Scalar heads**: `effort` (1-9), `distance`

**Lossless** (modular-only — see jxl-encoder#24)
- **Quality scale**: N/A (lossless); the picker is purely time-vs-bytes
- **Categorical axes**: `lz77_method` (None/RLE/Greedy/Optimal), `palette` (auto/no)
- **Scalar heads**: `tree_max_buckets` (16-256), `tree_num_properties` (3-16), `tree_sample_fraction` (0.1-0.7)
- **Objective**: **rd_time** (#56) — every scalar axis trades compute for compression. Bake with `--objective rd_time`.

### zenpng

- **Quality scale**: N/A (lossless)
- **Categorical axes**: `output_kind` (palette8 / rgb / rgba / gray / gray_a) — 5 cells. Mostly drives by `feat_distinct_color_bins` + `feat_alpha_present`; the picker is small.
- **Scalar heads**: `compression_effort` (0-30, zenflate range)
- **Rescue strategy default**: `KnownGoodFallback` (rgb + effort=15)

### zengif

- **Quality scale**: N/A; perceptual loss is dominated by palette quantization
- **Categorical axes**: `palette_size` (8/16/32/64/128/256), `dither` (none/floyd/sierra-lite)
- **Scalar heads**: `transparency_threshold` (0-128)
- **Rescue strategy default**: `KnownGoodFallback` (256 + sierra-lite)

### zenpicker (meta-picker)

- **Output**: 6 cells, one per `CodecFamily`
- **Quality scale**: zensim points
- **Categorical axes**: `family` (jpeg/webp/jxl/avif/png/gif)
- **Scalar heads**: none (purely categorical)
- **Bake metadata**: MUST include `zenpicker.family_order = "jpeg,webp,jxl,avif,png,gif"`. Validated at load via `MetaPicker::validate_family_order`.

### zensim V0_4

- **Different shape from pickers** — not a config selector. Inputs are pixel-pair features; output is one perceptual distance scalar.
- **n_outputs = 1**. No categorical cells, no hybrid heads.
- **Format is the same** (ZNPR v2, loaded via `zenpredict::Model::from_bytes`).
- **Training**: separate corpus (CID22 perceptual quality benchmark), separate trainer (lives in `zen/zensim/zensim-validate/src/mlp_train.rs`), separate audience (called by the codec's verify step, not picker step).
- **Re-bake triggers**: zenanalyze feature drift, V0_2 regression on the held-out CID22-val set.

---

## Default settings (the cheat sheet)

Set these unless you have a measured reason to deviate.

### Trainer (`tools/train_hybrid.py`)

| Flag | Default | Why |
|---|---|---|
| `--activation` | **`leakyrelu`** | sklearn's `MLPRegressor` with ReLU is single-threaded and 5–15× slower than LeakyReLU on our typical (74-input × 128 × 3 × 14k-row) workload. PR #57 surfaces this. |
| `--objective` | `size_optimal` | mean log-bytes loss. `zensim_strict` for SLA-bound traffic; `rd_time` for compute-bound traffic. |
| `--time-loss-weight` | `0.5` (when `--objective rd_time`) | balances time-head signal against bytes-head dominance. |
| `--reach-threshold` | `0.99` | strict default for `zensim_strict`. Codec consumers can re-threshold at request time via `threshold_mask`. |
| Hidden-layer sizing | `--hidden 192,192,192` for ≥30 features; `--hidden 128,128` for fewer | Capacity sweep (#zenanalyze internal) shows depth helps more than width past ~50 cross-termed inputs. |

### Bake (`tools/bake_picker.py`)

| Flag | Default | Why |
|---|---|---|
| `--dtype` | `f16` | Halves model size at ~no accuracy cost. Use `f32` only when round-trip diagnostics need bit-exactness; `i8` only when binary-size pressure is real. |
| `--allow-unsafe` | `false` (must be opted in) | Default refuses to bake when `safety_report.passed=false`. |
| `--no-manifest` | `false` | Legacy sibling `manifest.json` still emitted by default for codecs that haven't migrated to the in-bin metadata path. |

### Sweep harness (per-codec)

| Setting | Default | Why |
|---|---|---|
| Image sizes | `[64, 256, 1024, 0]` (= tiny, small, medium, large/native) | Per the four-dimension discipline. |
| Quality grid | `[0..70 step 5] + [70..100 step 2]` | Dense at high q; q step 5 from 0..70 catches structural failures at low q. |
| Per-cell sample size | ≥1 image × all sizes; aggregate ≥30 cells per `(size, zq)` band | Below the floor → DATA_STARVED_SIZE violation. |
| `effective_max_zensim` column | required | UNCAPPED_ZQ_GRID violation when missing at high zq. |
| `encode_ms` column | required | Even if the bake doesn't use `--objective rd_time` today, it might tomorrow; collect the data once. |

### Runtime (codec-side)

| Setting | Default | Why |
|---|---|---|
| `Model::from_bytes_with_schema(...)` | always use this | Hard-fail on stale bake. |
| `#[repr(C, align(16))]` wrapper around `include_bytes!` | always | Required for zero-copy aligned slices. |
| OOD bounds check on every encode | always when bounds present | Cheap; routes pathological inputs to fallback. |
| Reach gate threshold for `Profile::Strict` | `0.99` | Per-request callable via `threshold_mask`. |
| Two-shot rescue threshold | `RescuePolicy::default().rescue_threshold = 3.0` | Codec overrides per its quality-scale calibration. |

---

## Validation gates (block release)

A bake doesn't ship until **every** check passes:

1. **Trainer safety report**: `safety_report.passed = true` — no
   `PER_SIZE_TAIL`, `DATA_STARVED_SIZE`, `UNCAPPED_ZQ_GRID`,
   `OOD_LEAKAGE`.
2. **Held-out p99 zensim shortfall** below `1 pp` (size_optimal) /
   `0.5 pp` (zensim_strict).
3. **Round-trip check**: `tools/bake_roundtrip_check.py` matches
   numpy reference within tolerance for the bake's `--dtype`.
4. **Size invariance probe**: `tools/size_invariance_probe.py`
   passes per-`(image, target_zq)` cell-stability across sizes
   (≥ 90 % stable picks per cell).
5. **Schema hash committed** to a tracking issue (see #41) so the
   inventory of "what's deployed where" stays accurate.
6. **`feature_bounds` populated** in the bin (every codec must have
   the OOD path active; absent bounds are a release block).

For shipped consumer bumps:

7. **Pick-stability gate**: ≥ 90 % pick agreement vs the previously
   shipped bake on a held-out web-traffic-representative corpus
   (≥ 100 images mixing photo + screen + line-art).

---

## Known landmines

Lessons learned the hard way; document and ward against, don't
re-discover.

### "I'll bake at the new budget, picker calibration handles itself"

No. Re-bake every consumer when the `pixel_budget` default moves.
`feat_patch_fraction_fast` drifts ~0.1 σ across the 0.5 → 4 MP range
(#48). The picker's internal scaling is anchored to a specific
budget; using a model trained at one budget with features sampled at
another is a calibration mismatch, full stop.

### "The picker's prediction is fine; we don't need OOD"

It's not fine for OOD inputs. The MLP extrapolates linearly past the
trained envelope and produces confident-looking garbage. **Every
codec MUST run `first_out_of_distribution` before argmin** and
fall through to a known-good config when an input is out of range.

### "Adding a new training corpus class doesn't need a re-bake"

It does. Adding screen-content images to a corpus that was 80 %
photographic shifts the trained `(feature mean, feature variance)`,
which shifts the StandardScaler, which means the same image's
features get scaled differently by the runtime. Always re-bake when
the training-corpus composition changes.

### "Composite features are stable enough to default-on"

(#6.) No. `TextLikelihood` / `ScreenContentLikelihood` /
`NaturalLikelihood` / `LineArtScore` are hand-tuned weighted
combinators of stable raw signals. The combinator coefficients drift
as the calibration corpus matures. Keep them behind the
`composites` cargo feature — stays opt-in until calibration is
ratified by a multi-quarter corpus stability check (see #6).

### "Bake metadata can be approximate / left for later"

The metadata blob is the ONLY way the codec gets bake-time data
(reach rates, calibration metrics, feature_columns,
hybrid_heads_layout, family_order). A bake without metadata can't be
safely consumed at runtime. Always populate it; the trainer emits
all required keys by default.

### "If the picker's pick is bad we can fix it post-hoc"

You can't. By the time the bytes are on the wire the pick decision
is already amortized into perceived encode quality. The two-shot
rescue path is the LAST line of defense; don't skip the OOD bounds
check, the reach gate, or the schema_hash gate "to save a few µs".

### "I'll plan to support a new metric (butter / ssim2) later — picker design today should anticipate it"

You can't really anticipate it; metric-specific calibration is the
work, not the wiring. Ship `--objective size_optimal` with zensim,
emit the encode-time + zensim columns in the sweep TSV, and when
butter (or ssim2) becomes a real consumer requirement, run the same
trainer on a butter-evaluated corpus to produce `picker_butter.bin`
side-by-side with the existing zensim bake. The runtime
already supports loading multiple bakes per codec (`PickerProfile`
selects at session start); see [`SAFETY_PLANE.md`](SAFETY_PLANE.md).

---

## Cross-references

- Tutorial: [`FOR_NEW_CODECS.md`](FOR_NEW_CODECS.md)
- Codec-side rescue + safety plane: [`SAFETY_PLANE.md`](SAFETY_PLANE.md)
- Runtime API: [`../zenpredict/README.md`](../zenpredict/README.md)
- Meta-picker: [`../zenpicker/README.md`](../zenpicker/README.md)
- Migration from the unpublished zenpicker Rust shell:
  [`../MIGRATION.md`](../MIGRATION.md)
- Project-wide sweep discipline:
  `~/work/claudehints/CLAUDE.md` → "Sweep / Calibration /
  Source-informing Benchmark Discipline"

## Tracking — open work informing this document

| Issue / PR | Topic | State |
|---|---|---|
| [#41](https://github.com/imazen/zenanalyze/issues/41) | Track downstream FeatureSet shapes | open |
| [#43](https://github.com/imazen/zenanalyze/issues/43) | time-budgeted training objective (compute-trade knobs) | open |
| [#44](https://github.com/imazen/zenanalyze/pull/44) | raise `pixel_budget` to 2 MP | open — gated on re-bake of shipped pickers per #48 |
| [#46](https://github.com/imazen/zenanalyze/issues/46) | adaptive per-image budget | open |
| [#47](https://github.com/imazen/zenanalyze/issues/47) | expand budget-research corpus | open |
| [#48](https://github.com/imazen/zenanalyze/issues/48) | `patch_fraction_fast` doesn't saturate vs budget | open |
| [#49](https://github.com/imazen/zenanalyze/issues/49) | per-feature min-sample floor | landed (`9745949`) |
| [#51](https://github.com/imazen/zenanalyze/issues/51) | per-(image, size) zensim ceiling | open — PR #52 in flight |
| [#52](https://github.com/imazen/zenanalyze/pull/52) | ceiling-aware ZQ filter + UNCAPPED_ZQ_GRID gate | open |
| [#53](https://github.com/imazen/zenanalyze/issues/53) | dynamic dispatch tree | partial — PR #54 ships stages 0+1.5 |
| [#54](https://github.com/imazen/zenanalyze/pull/54) | `analyze_with_dispatch_plan` stages 0+1.5 | open |
| [#55](https://github.com/imazen/zenanalyze/issues/55) | scorer-composed argmin | landed (`76bc5a8`) |
| [#56](https://github.com/imazen/zenanalyze/issues/56) | `--objective rd_time` + time head | open |
| [#57](https://github.com/imazen/zenanalyze/pull/57) | bake_picker finite-bound sentinels + leakyrelu doc | open |

This table is the live map of what's in flight. Update when an
item lands or a new one is filed.
