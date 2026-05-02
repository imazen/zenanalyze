# Picker safety-default path + multi-objective runtime, 2026-05-02

> **Revised after user pushback (~04:30 UTC).** Earlier draft proposed
> a CATALOG of objective-specific MLPs (Default / RdSlope /
> MetricTimeMasked, separate trains, separate bakes). The user
> correctly noted (a) safety masking applies uniformly across
> objectives — Default's safety story carries through, (b) shipping
> more than 2 picker bakes per (codec, metric) is heavy from
> storage / operations / mental-model standpoints. Revised plan:
> **one MLP per (codec, metric)** with multi-output heads always
> emitted; **three objectives implemented at RUNTIME via score
> functions + masks**, not separate models. Same safety story,
> ~3× fewer bakes, ~5 days less engineering.

User-driven architectural pivot: safety should be the **default
path**, not opt-in. Every encode runs input-OOD → predict →
output-plausibility → mask+score → confidence-gap → param-clamp →
fallback as a single atomic decision; codecs receive a structured
`Pick` result, not raw model outputs. **Three objectives (Default,
RdSlope, MetricTimeMasked) ride on the same MLP — implemented as
runtime masks and scoring functions over the model's
already-emitted multi-output heads.**

This doc is the architectural target. Cross-references the
[`orchestration_architecture_2026-05-02.md`](orchestration_architecture_2026-05-02.md)
substrate plan and the
[`audits-2026-05-02/zenpicker_zenpredict.md`](audits-2026-05-02/zenpicker_zenpredict.md)
runtime audit.

## Current state (more shipped than the audit suggested)

Already in main / shipped infrastructure:

- **Multi-output hybrid heads**: trainer emits `bytes_log[N_CELLS]`,
  optional `time_log[N_CELLS]` (gated `--emit-metric-head`),
  optional `metric_log[N_CELLS]`.
- **`--time-budget-multiplier` label filter**: per-cell candidates
  filtered to `time_ms ≤ baseline_ms[size_class] × multiplier`
  before the min-bytes pick. `BUDGET_INFEASIBLE` safety gate fires
  when too many (image, size) pairs go over budget at every zq.
- **Input-OOD + output-OOD bounds**: `feature_bounds` (top-level
  Section, p01/p99 per feature column) and `output_bounds`
  (TLV metadata, p01/p99 per output dim). Both checked at every
  encode.
- **Rescue scaffolding**: `RescuePolicy::from_bake`,
  `RescueStrategy::KnownGoodFallback`, `cell_rescue_hints`,
  `zq_fallback_table` — a fairly complete two-shot rescue taxonomy.
- **Score recipes documented** in `PRINCIPLES.md` (Default, Hard
  time cap, Soft RD-vs-time, Time-to-percent-saved, Multi-metric).
- **`argmin_masked_with_scorer`** — caller composes any score from
  hybrid heads via a closure.

Gaps:

1. **No multi-MLP-per-objective architecture.** Today: one MLP per
   metric (PRINCIPLES.md §"Metric choice — Option A"). The user
   wants per-objective bakes too.
2. **No `ParamClamp` rescue strategy** — only `KnownGoodFallback`
   (full replacement). When an output scalar overshoots its
   safe range, today's only option is to discard the entire pick.
3. **No default-path orchestration in `zenpredict`** — codec composes
   the safety chain manually each call. The PRINCIPLES.md §3 line
   "Codecs MUST handle the `None` case via either
   `first_out_of_distribution + RescueStrategy::KnownGoodFallback`
   or explicit `unwrap_or(default)`" admits this.
4. **No production codec uses any of this** (audit finding):
   zenwebp uses Default recipe only; zenjpeg/zenjxl/zenavif don't
   load pickers at all.

## Single-MLP, multi-objective architecture

**One MLP per (codec, metric).** The MLP always emits all three
output heads (default-on, no longer opt-in via
`--emit-metric-head`):

- `bytes_log[N_CELLS]` — log-bytes per cell representative config
- `time_log[N_CELLS]` — log-encode-ms per cell representative
- `metric_log[N_CELLS]` — log-achieved-metric per cell representative
- `scalars[N_CELLS][n_scalar_axes]` — per-cell continuous knob
  predictions (existing trainer behavior)

Per cell, the "representative" is the config minimizing bytes among
those reaching `target_zq` — the existing Default-trained label.

**Three objectives at runtime via masks + score functions:**

| Objective | Mask `m[c]` | Score | Argmin |
|---|---|---|---|
| `Default { target_zq }` | `true` | `bytes_log[c]` | min score |
| `RdSlope { target_zq, μ }` | `true` | `bytes_log[c] + μ × time_log[c]` | min score |
| `MetricTimeMasked { target_zq, T, N_ms }` | `metric_log[c] ≥ log(T)` ∧ `time_log[c] ≤ log(N_ms)` | `bytes_log[c]` | min score; infeasible-sentinel if mask is empty |

All three run through the same `pick()` orchestrator + safety
chain. The MLP doesn't care what objective it's called with — the
model outputs are objective-agnostic predictions of per-cell
properties.

**Catalog (hybrid)**:

```
codec_crate/src/pickers/
  picker_<metric>.bin           # base — Default + MetricTimeMasked via runtime mask
  picker_rdslope_<metric>.bin   # OPTIONAL — slope-tuned, μ as continuous input
```

- **Base bake**: always shipped. One MLP per (codec, metric) that
  emits all multi-output heads. Serves Default and MetricTimeMasked
  via runtime composition.
- **Slope-tuned bake**: per-codec opt-in. Trains a separate MLP
  with μ as a continuous input feature (normalized to the codec's
  `mu_range_for_codec`); argmin labels per `(image, μ)` minimize
  `bytes + μ × time_ms`. Within-cell scalars are slope-aware.
  Codecs where time/RD tradeoff matters in production (AVIF, JXL,
  zenwebp at high effort) ship it; codecs where it doesn't (zenjpeg
  is fast enough that μ rarely matters) skip.

`Predictor::pick(features, PickerObjective::RdSlope { μ })`:
1. Try to load slope-tuned bake; if present, use it directly with
   μ as input.
2. If absent, fall back to base bake + runtime composition
   (`bytes_log + μ × time_log`) with the within-cell-scalar caveat
   noted.

Storage:

| Layout | 4 codecs × 2 metrics | Disk |
|---|---|---|
| Base only (every codec ships) | 8 bakes | ~1.2 MB |
| + slope-tuned for AVIF + JXL + WebP | 14 bakes | ~2.1 MB |
| Worst case (all codecs ship slope-tuned) | 16 bakes | ~2.4 MB |

Within the "more than 2 picker bakes per (codec, metric) is heavy"
budget — even worst case is 2 bakes per (codec, metric).

### Trade-off vs the per-objective-MLP design

The single-MLP design uses one model's predictions and composes the
objective at runtime. Accuracy difference is concentrated at the
**within-cell scalar setpoint**: each cell's predicted continuous
knobs (`speed`, `sns_strength`, …) are calibrated for the min-bytes
representative config. When `RdSlope { high μ }` runs at runtime,
the cell pick correctly favors fast cells, but the scalars *within*
the chosen cell are still min-bytes-tuned — slope-suboptimal at the
edges.

For **Default** and **MetricTimeMasked** the loss is bounded:

- Cells bound the major decisions (e.g., zenwebp's `method ×
  segments`); within-cell scalar choices trade off less than
  between-cell choices.
- MetricTimeMasked is a *hard* constraint (no continuous tradeoff
  coefficient); the runtime mask is exact.
- The dominant signal (cell choice) IS objective-aware via
  `bytes_log + μ × time_log` for slope or via the mask for
  MetricTimeMasked.

For **RdSlope** the loss is harder to bound because:

- Slope is a *continuous* tradeoff; the within-cell scalar error
  compounds with μ.
- A standalone RdSlope MLP can take μ as a continuous input
  feature, letting it learn "at low μ prefer slow-but-small;
  at high μ prefer fast-but-larger" as a function of μ — the
  shared MLP never sees μ at training time.
- Time predictions are intrinsically harder than bytes (CPU/memory/
  ILP variance); a slope-tuned model has a chance to learn which
  configs yield CONSISTENT time, not just the cell-representative
  time.

→ **Hybrid catalog**: ship one base MLP per (codec, metric) for
Default + MetricTimeMasked, plus an OPTIONAL slope-tuned MLP per
(codec, metric) for codecs where the RD/time tradeoff matters in
production (AVIF and JXL definitely; WebP at high effort; JPEG
probably not). See the "Catalog" section below.

## Safety-violation taxonomy (5 classes)

The default path runs all five checks on every encode, in order:

| # | Check | Existing infra | Action on violation |
|---|---|---|---|
| 1 | **Input OOD** — feature is outside p01/p99 of training corpus | `feature_bounds` + `first_out_of_distribution()` ✅ | `KnownGoodFallback` |
| 2 | **NaN feature** (sample-count floor; #49) | `result.get()` returns `None` ✅ | `KnownGoodFallback` |
| 3 | **Output OOD** — predicted output outside p01/p99 of training preds | `output_bounds` + `output_first_out_of_distribution()` ✅ | downgrade to lower confidence; if strict mode, `KnownGoodFallback` |
| 4 | **Confidence gap too narrow** — chosen and runner-up scores differ by < threshold | not yet implemented | `ParamClamp` to runner-up's "safer" axis values, OR fall back to `Default` MLP if running RdSlope/MetricMasked |
| 5 | **Output param clamp** — predicted scalar (e.g., `speed`, `sns_strength`) outside the codec's hard range | not yet implemented | `ParamClamp` to nearest legal value |

## Fallback taxonomy (two strategies)

### Strategy `ParamClamp`

Keep most of the picker's output, clamp specific axes to safe
ranges. Used when one or two scalar outputs are slightly out of
range but the categorical pick is reasonable.

Examples:
- Picker says `speed=11`, codec range is `[1, 10]` → clamp to `10`.
- Picker confidence gap on cell choice is too narrow → take cell
  pick, but use runner-up's scalar values (more conservative).
- Picker says `sns_strength=120`, codec range `[0, 100]` → clamp.

Provenance: caller sees `Pick::Clamped { config_id, scalars,
clamped_axes: ["speed", "sns_strength"], reasons: [...] }`.

### Strategy `KnownGoodFallback` (full replacement)

Discard the picker's output entirely; use a baked-in baseline
config from `zq_fallback_table`. Used when:
- Input OOD or NaN feature
- Output OOD (in strict mode)
- Picker output is structurally invalid (cell id out of range,
  scalar NaN)

Provenance: `Pick::Replaced { fallback_config, fallback_reason }`.

## Default-path runtime contract

`zenpredict` ships ONE convenience method per objective that runs
the full safety chain. Caller doesn't compose:

```rust
pub enum PickerObjective {
    Default { target_zq: f32 },
    RdSlope { target_zq: f32, mu_bytes_per_ms: f32 },
    MetricTimeMasked { target_zq: f32, target_metric: f32, time_cap_ms: f32 },
}

pub enum Pick {
    HappyPath {
        config_id: u32,
        scalars: Vec<f32>,
        confidence_gap: f32,
    },
    Clamped {
        config_id: u32,
        scalars: Vec<f32>,
        clamped_axes: Vec<&'static str>,
        reasons: Vec<ClampReason>,
    },
    Replaced {
        fallback_config: u32,
        fallback_scalars: Vec<f32>,
        reason: ReplaceReason,
    },
}

impl Predictor {
    pub fn pick(
        &self,
        features: &[f32],
        objective: PickerObjective,
    ) -> Result<Pick, ZenpredictError>;
}
```

The codec calls `pick()` once. All 5 safety checks run; the
returned `Pick` enum tells the codec what was applied.
Observability is structured (every variant carries reasons).

## Trainer responsibilities

To support the default path, the trainer always emits all three
output heads (no longer opt-in) and bakes additional metadata per
`.bin`:

| TLV key | Type | Purpose | Status |
|---|---|---|---|
| `param_clamp_ranges` | `(f32, f32)[n_scalar_outputs]` | hard min/max per scalar; runtime clamps to these | new in D2 |
| `confidence_gap_threshold` | f32 | minimum gap between top-2 scores; below = trigger downgrade | new in D2 |
| `mu_range_for_codec` | `(f32, f32)` | per-codec normalization for the μ input (e.g., `(0.0, 0.05)` bytes/ms for AVIF) | new in D2 |
| `time_calibration_ms_per_mp` | f32 | median ms-per-MP at training; runtime measures local CPU and applies multiplier | new in D3 |
| `output_bounds` | already there | per-output p01/p99 | shipped |
| `feature_bounds` | already there | per-feature p01/p99 | shipped |
| `zq_fallback_table` | already there | fallback config per zq band | shipped |
| `safety_compact` + `cell_rescue_hints` | already there | RescuePolicy substrate | shipped |

Note: no `picker_objective` TLV — one bake supports all three
objectives, selected at runtime via the `PickerObjective` enum
passed to `pick()`.

The trainer gates the bake on:
- **Fallback exercisable**: every `(image, size, zq)` cell that
  triggers a fallback in the safety report must route to a
  config in `zq_fallback_table` that's been observed in the
  pareto sweep (no hallucinated fallback configs).
- **Param clamps non-empty**: every scalar output has a
  `(min, max)` range derived from the training corpus.
- **Confidence gap calibrated**: `confidence_gap_threshold` set
  to a value that correctly classifies the safety-report's
  flagged "narrow win" cells.

CI gate: bake fails to publish if any of these three conditions
isn't met.

## Storage / operational model

Per codec, the picker catalog lives at:
```
codec_crate/src/pickers/
  picker_default_zensim.bin
  picker_default_butter.bin
  picker_rd_slope_zensim.bin
  picker_rd_slope_butter.bin
  picker_metric_masked_zensim.bin
  picker_metric_masked_butter.bin
```

OR (preferred per the substrate-pivot doc) in R2:
```
r2://pickers.imazen/zenwebp/{codec_commit}/picker_{obj}_{metric}.bin
```

with embedded baseline `picker_default_zensim.bin` in the codec
crate as cold-cache fallback.

Codec at runtime selects:

```rust
let bin = pickers.load(PickerObjective::RdSlope { .. }, Metric::Zensim)?;
let pick = bin.pick(&features, objective)?;
```

## Phased plan (revised: D3+D4 collapse into runtime work)

**~8 days total**, down from the per-objective-MLP plan's ~13 days.
Single-MLP-multi-objective architecture eliminates the separate
training passes for RdSlope and MetricTimeMasked; both become
runtime score+mask functions over the existing multi-output heads.

### Phase D1 — Default-path orchestration in zenpredict (~3 days)

Add `Pick` enum, `PickerObjective` enum, `Predictor::pick()`
wrapper. Wire input-OOD + output-OOD + objective-specific
mask+score + cell confidence-gap + param-clamp + KnownGoodFallback
into one method. Keep the existing `argmin_masked_with_scorer` for
advanced callers (escape hatch).

| # | Item | Effort |
|---|---|---:|
| D1.1 | `Pick` + `PickerObjective` + `ClampReason` + `ReplaceReason` enums | 4 hr |
| D1.2 | `Predictor::pick()` orchestrator with mask+score per objective | 1 day |
| D1.3 | Param-clamp logic + `param_clamp_ranges` TLV reader | 4 hr |
| D1.4 | Confidence-gap check + `confidence_gap_threshold` TLV reader | 4 hr |
| D1.5 | Tests: synthetic feature → each `Pick` variant fires for each objective | 1 day |

### Phase D2 — Trainer bakes extra TLV + heads-default-on (~2 days)

| # | Item | Effort |
|---|---|---:|
| D2.1 | Make `time_log` + `metric_log` heads default-on (remove `--emit-metric-head` opt-in) | 2 hr |
| D2.2 | Compute `param_clamp_ranges` from corpus (per scalar axis: p01/p99 of legal range × clamp tolerance) | 2 hr |
| D2.3 | Compute `confidence_gap_threshold` from safety report's narrow-win cells | 2 hr |
| D2.4 | Emit `mu_range_for_codec` TLV (per-codec μ calibration range, normalized to `[0,1]`) | 1 hr |
| D2.5 | CI gate: bake fails if (a) any safety-flagged cell can't route via `zq_fallback_table`, (b) any scalar lacks clamp range, (c) confidence threshold doesn't classify narrow wins | 1 day |

### Phase D2b — Slope-tuned MLP training mode (~1 day)

| # | Item | Effort |
|---|---|---:|
| D2b.1 | `--objective rd-slope` flag in train_hybrid.py | 30 min |
| D2b.2 | μ as continuous input feature (sample over `mu_range_for_codec` during training; one input dim) | 4 hr |
| D2b.3 | Per-(image, μ) argmin label assignment: minimize `bytes + μ × time_ms` over reachable configs | 2 hr |
| D2b.4 | Output filename convention: `picker_rdslope_<metric>.bin` (vs `picker_<metric>.bin` for the base bake) | 30 min |
| D2b.5 | Picker config knob: `EMIT_SLOPE_TUNED_BAKE: bool` per codec (default False except AVIF/JXL/zenwebp-effort) | 30 min |

### Phase D3 — Time calibration (~1 day)

| # | Item | Effort |
|---|---|---:|
| D3.1 | `time_calibration_ms_per_mp` TLV (median ms-per-MP at training) | 2 hr |
| D3.2 | Startup CPU multiplier probe (encode a fixed reference image, compare to baked baseline) | 4 hr |
| D3.3 | RdSlope μ correction: `μ_effective = μ_caller × cpu_multiplier` | 2 hr |

Closes issue #56 (RD-vs-time bake calibration protocol).

### Phase D4 — Production wiring per codec — base bake (~2 days)

Wire each codec to call `pick()` with the base bake. Default
objective: `Default { target_zq }`. Codec exposes
`with_objective(o)` knob for callers who want RdSlope or
MetricTimeMasked.

| # | Item | Effort |
|---|---|---:|
| D4.1 | zenwebp: replace existing `argmin_masked_with_scorer` calls with `pick()` | 4 hr |
| D4.2 | zenjpeg: load picker.bin via `pick()` (closes issue #128) | 4 hr |
| D4.3 | zenjxl, zenavif: same pattern | 8 hr |
| D4.4 | Integration tests per codec (Default + MetricTimeMasked produce sensible picks; RdSlope falls back to runtime composition with caveat note) | 1 day |

D4 overlaps significantly with the rapid-iteration plan's Phase 0
(wire the 3 idle pickers).

### Phase D5 — Slope-tuned bakes per opt-in codec (~3 days)

Per-codec slope-tuned MLP shipping. Each codec independently
decides whether to ship the slope-tuned variant. `pick()` already
falls back to runtime composition when the slope-tuned bake is
absent (D1 plumbing).

| # | Item | Effort |
|---|---|---:|
| D5.1 | Train + bake `picker_rdslope_zensim.bin` for zenavif (rav1e is the slowest; biggest slope impact) | 6 hr |
| D5.2 | Same for zenjxl | 6 hr |
| D5.3 | Same for zenwebp at high-effort regime | 6 hr |
| D5.4 | Operator-side switching: `pick(RdSlope { μ })` automatically picks slope-tuned bake when present | 2 hr |
| D5.5 | Smoke test: each codec's slope-tuned bake produces measurably better picks at extreme μ vs base+composition fallback | 1 day |

zenjpeg skipped — encode is fast enough that μ rarely shifts the
optimal config (the within-cell scalar loss is in the noise floor
of zenjpeg's fast-encoder regime).

## Trade-offs / open questions

1. **Within-cell scalar slope-suboptimality** (the trade-off we
   accepted): single-MLP scalar regressors are min-bytes-tuned, so
   `RdSlope { high μ }` picks the right cell but uses
   slightly-too-slow scalars within that cell. Magnitude probably
   small; if measurement shows it matters, add per-objective
   scalar heads (~3× scalar output dims) without changing
   MLP class.

2. **μ range per codec**: each codec has different
   bytes/ms-tradeoff regimes (zenjpeg fast, rav1e slow). The μ
   normalization range needs per-codec calibration. `mu_range_for_codec`
   TLV (Phase D2.4) handles this.

3. **(T, N) feasibility surface**: for `MetricTimeMasked`, the
   feasible set varies per image. Some images have NO cell whose
   `metric_log[c] ≥ T` AND `time_log[c] ≤ N` — `pick()` must
   reliably return `infeasible-sentinel` and route to fallback.

4. **Time-calibration drift**: a picker baked on a fast machine
   sees small `time_log` values; deployed on a slow machine, the
   `μ × time_log[c]` term under-weights time. The startup CPU
   multiplier probe (Phase D3.2) corrects this. Probe accuracy
   matters for slope correctness — needs a stable reference image.

5. **Backward compat**: existing `argmin_masked_with_scorer`
   stays as the escape hatch; the new `pick()` is purely additive.
   Codecs migrate at their own pace. zenwebp's existing usage of
   `argmin_masked_with_scorer` keeps working.

6. **Single bake per (codec, metric)**: 4 codecs × 2 metrics =
   8 bakes. PRINCIPLES.md §"Metric choice" Option A (one bake per
   metric) extends naturally — three objectives all ride on one
   model since the model outputs are objective-agnostic per-cell
   property predictions.

## Cross-references

- `orchestration_architecture_2026-05-02.md` — the substrate
  (R2 + vast.ai) where these bakes get produced and stored.
- `rapid_iteration_plan_2026-05-02.md` — Phase 0 codec wiring
  (3 idle pickers) is a prereq for Phase D5.
- `audits-2026-05-02/zenpicker_zenpredict.md` — the runtime
  audit that motivated the default-path design.
- `audits-2026-05-02/zentrain_zenanalyze.md` — the trainer-side
  audit that motivated the per-objective MLP design.
- `zentrain/PRINCIPLES.md` §"Score recipes" — current closure-based
  approach this plan supersedes; §"OOD bounds — input AND
  output, in every bake" — already-shipped infra this plan
  builds on.
