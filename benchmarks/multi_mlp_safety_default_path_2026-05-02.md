# Multi-MLP picker architecture + safety-default path, 2026-05-02

User-driven architectural pivot: pickers should ship as a **catalog
of objective-specific MLPs** rather than one multi-output MLP +
caller-composed scorer closures. Safety should be the **default
path**, not opt-in. Every encode runs input-OOD → predict →
output-plausibility → confidence-gap → fallback as a single atomic
decision; codecs receive a structured pick result, not raw model
outputs.

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

## Three-MLP architecture

Per codec × per objective × per metric. Three objectives ship as
default; each can take any supported metric:

### MLP class A — `Default` (smallest bytes at target zq)

What we have today. Argmin label per (image, target_zq):
config minimizing `bytes` among configs reaching `target_zq`.
Inputs: features + `target_zq`. Outputs: `bytes_log[N_CELLS]`.

### MLP class B — `RdSlope` (μ-aware bytes/ms tradeoff)

Argmin label per (image, μ): config minimizing `bytes + μ × time_ms`.
Inputs: features + `target_zq` + `μ` (continuous, normalized to
`[0, 1]` over a sensible per-codec range, e.g., `0` to `0.05 bytes/ms`
for AVIF). Outputs: `bytes_log[N_CELLS]` + `time_log[N_CELLS]`.

Trained against the SLOPE, not the multi-output composition.
Calibration: per-platform `time_calibration_ms_per_mp` baked in
metadata + startup-measured per-CPU multiplier (issue #56).

### MLP class C — `MetricTimeMasked` (metric ≥ T AND time ≤ N)

Argmin label per (image, T, N): config minimizing `bytes` among
configs satisfying `metric ≥ T` AND `time ≤ N`. Inputs: features +
`target_zq` + `target_metric_T` + `time_cap_N`. Outputs:
`bytes_log[N_CELLS]` + `metric_log[N_CELLS]` + `time_log[N_CELLS]`.

Returns "infeasible" (sentinel cell id) when no config satisfies
both constraints — caller routes to fallback.

### Catalog

Per codec, the bake catalog grows from "one bake per metric" to:

```
picker_default_<metric>.bin       # smallest bytes at target_zq
picker_rd_slope_<metric>.bin      # μ-aware
picker_metric_masked_<metric>.bin # ≥T, ≤N
```

For 3 metrics × 3 objectives × 4 codecs: 36 bakes × ~150 KB =
~5 MB total. Manageable storage; trivial selection at runtime
(pick `.bin` based on caller's `PickerObjective` enum).

The trade-off vs one-MLP-many-closures:

| Metric | Multi-output MLP + closure | Per-objective MLP |
|---|---|---|
| Argmin accuracy | model approximates each objective from the same logits | each model fully calibrated for its objective |
| Storage | 1× per metric | 3× per metric |
| Bake cost | 1× per metric | 3× per metric (most cost is encode/zensim, shared across objectives — only the trainer step duplicates) |
| Cross-bake bias | possible (model favors the head it sees gradient on most) | impossible by construction |
| Caller surface | closure boilerplate | enum select |

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

To support the default path, the trainer bakes additional
metadata per `.bin`:

| TLV key | Type | Purpose |
|---|---|---|
| `picker_objective` | u8 enum | `Default` / `RdSlope` / `MetricTimeMasked` |
| `param_clamp_ranges` | `(f32, f32)[n_scalar_outputs]` | hard min/max per scalar; runtime clamps to these |
| `confidence_gap_threshold` | f32 | minimum gap between top-2 scores; below = trigger downgrade |
| `output_bounds` | already there | per-output p01/p99 |
| `feature_bounds` | already there | per-feature p01/p99 |
| `zq_fallback_table` | already there | fallback config per zq band |
| `time_calibration_ms_per_mp` | f32 | per-MP encode-time baseline at training; runtime measures local CPU and applies multiplier |

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

## Phased plan

Roughly 3–4 weeks of focused engineering, in five phases. Each
phase delivers something useful by itself.

### Phase D1 — Default-path orchestration in zenpredict (~3 days)

Add `Pick` enum, `PickerObjective` enum, `Predictor::pick()`
wrapper. Wire input-OOD + output-OOD + cell confidence-gap +
param-clamp + KnownGoodFallback into one method. Keep the
existing `argmin_masked_with_scorer` for advanced callers.

| # | Item | Effort |
|---|---|---:|
| D1.1 | `Pick` + `PickerObjective` + `ClampReason` + `ReplaceReason` enums | 4 hr |
| D1.2 | `Predictor::pick()` orchestrator | 1 day |
| D1.3 | Param-clamp logic + `param_clamp_ranges` TLV reader | 4 hr |
| D1.4 | Confidence-gap check + `confidence_gap_threshold` TLV reader | 4 hr |
| D1.5 | Tests: synthetic feature → each Pick variant fires | 1 day |

### Phase D2 — Trainer bakes extra TLV (~2 days)

| # | Item | Effort |
|---|---|---:|
| D2.1 | Compute `param_clamp_ranges` from corpus | 2 hr |
| D2.2 | Compute `confidence_gap_threshold` from corpus | 2 hr |
| D2.3 | Emit `picker_objective` TLV | 1 hr |
| D2.4 | Validate fallback exercisable (CI gate) | 4 hr |
| D2.5 | `time_calibration_ms_per_mp` (issue #56) | 1 day |

### Phase D3 — `RdSlope` MLP (~3 days)

| # | Item | Effort |
|---|---|---:|
| D3.1 | `--objective rd-slope` flag in train_hybrid.py; per-(image, μ) label assignment | 1 day |
| D3.2 | μ as continuous input feature (normalized) | 4 hr |
| D3.3 | Trainer emits `bytes_log + time_log` (already does); validate | 4 hr |
| D3.4 | Calibration probe: startup-measured CPU multiplier | 4 hr |
| D3.5 | Smoke test on zenwebp: bake with `--objective rd-slope`, run `Predictor::pick(RdSlope { μ })`, confirm trades bytes for ms | 1 day |

### Phase D4 — `MetricTimeMasked` MLP (~3 days)

Same shape as D3, but argmin under (T, N) constraints. Returns
infeasible-sentinel when no config qualifies.

| # | Item | Effort |
|---|---|---:|
| D4.1 | `--objective metric-masked` flag; (T, N)-constrained label assignment | 1 day |
| D4.2 | T, N as continuous inputs | 4 hr |
| D4.3 | Infeasible-sentinel handling at predict time | 4 hr |
| D4.4 | Smoke test on zenwebp | 1 day |

### Phase D5 — Production wiring (~2 days)

Wire one objective per codec to actually load the right bake at
runtime, validate the default path produces sane picks. Closes
the "trained models that don't run in production" gap from the
audits.

| # | Item | Effort |
|---|---|---:|
| D5.1 | zenwebp picks `Default zensim` by default; expose `with_objective()` knob | 4 hr |
| D5.2 | zenjpeg loads picker.bin via `pick()` | 4 hr |
| D5.3 | zenjxl, zenavif similarly | 8 hr |
| D5.4 | Integration tests per codec | 1 day |

## Trade-offs / open questions

1. **Catalog explosion**: 4 codecs × 3 objectives × 3 metrics =
   36 bakes. Each ~150 KB. Storage: trivial. Bake compute:
   3× the trainer work (sweep is shared). Operational: codec
   needs a small "pickers" registry + selection logic.

2. **μ range per codec**: each codec has different
   bytes/ms-tradeoff regimes (zenjpeg fast, rav1e slow). The μ
   normalization range needs per-codec calibration. Bake into
   metadata.

3. **(T, N) feasibility surface**: for `MetricTimeMasked`, the
   feasible set varies per image. Some images have NO config
   that satisfies (high T, low N) — picker must reliably
   detect this and return infeasible-sentinel. Trainer needs to
   include "infeasible" as a label class.

4. **MLP architectural cost**: 3× MLPs means 3× training time;
   not a problem at our corpus sizes (~minutes per train) but
   matters when we add more codecs or metrics.

5. **Backward compat**: existing `argmin_masked_with_scorer`
   stays; the new `pick()` is purely additive. Codecs migrate at
   their own pace. zenwebp's existing usage of
   `argmin_masked_with_scorer` keeps working.

6. **Catalog vs single-bake hybrid**: an alternative is one bake
   per metric with all three objectives' label heads. Simpler
   storage; harder to argue calibration purity. PRINCIPLES.md
   §"Metric choice" already chose Option A (one bake per metric)
   for that reason. Per-objective bakes follow the same logic.

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
