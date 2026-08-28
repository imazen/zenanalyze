# Meta-picker per-family degradation — measured tables + the API decision (#85)

Status of imazen/zenanalyze#85 as of 2026-08-28, written against the current
source (`zenpicker/src/{lib,route}.rs`, `zenpredict/src/{argmin,encode_strategy}.rs`).

## What is landed

| piece | where | state |
|---|---|---|
| Feasibility mask (`cost(family, min-effort) > budget` → drop) | `AllowedFamilies::viable(mode, latency_ms, per_family_est_ms)` (`zenpicker/src/lib.rs`), used by `MetaPicker::route` and `family_rule` | shipped |
| `EncodeBudget` / `EncodeMode` / `PickerStrategy` | `zenpredict/src/encode_strategy.rs` | shipped |
| `argmin_masked(scores, mask, transform, offsets)` with `ArgminOffsets { uniform, per_output }` | `zenpredict/src/argmin.rs`, `Predictor::argmin_masked` | shipped; `MetaPicker::pick` passes `None` offsets |
| `feature_transforms` padded to `n_inputs` at bake | `cf6972e` | shipped |
| Python cargo fallbacks name the owning package | `1458f71` | shipped |
| **Per-family degradation + per-effort cost tables** (this session) | `zentrain/tools/fit_family_degradation.py` → `benchmarks/family_degradation_{zenwebp,zenjxl,zenavif}_2026-08-28.{tsv,md}` | measured, 3 of 4 lossy families (see below) |

## What the tables say (canonical 2026-06-27, `score_zensim`, ln-bytes penalty vs the reference effort)

`δ(cap) = ln bytes_at(cap) − ln bytes_at(ref)`, `bytes_at(E)` = smallest encode reaching
the target with effort ≤ E (any cell / q). Pooled over sizes; per-size rows are in the TSVs.

| family (ref) | cap | target 50 | target 70 | target 85 | target 95 |
|---|---|---:|---:|---:|---:|
| webp (m6) | ≤ m4 | +8.2 % | +6.6 % | +3.7 % | +1.0 % |
| webp (m6) | ≤ m2 | +13.3 % | +10.4 % | +6.6 % | +2.7 % (reach 80 %) |
| jxl (e9) | ≤ e7 | +4.4 % | +0.7 % | +0.4 % | 0 % |
| jxl (e9) | ≤ e5 | +13.7 % | +4.7 % | +8.3 % | unreachable (0 %) |
| jxl (e9) | ≤ e3 | +20.2 % | +6.5 % | +15.4 % | unreachable |
| jxl (e9) | ≤ e2 | +169 % | +108 % | +73 % | unreachable |
| avif (s2) | ≥ s4 | +5.7 % | +3.9 % | +3.7 % | +1.6 % |
| avif (s2) | ≥ s6 | +12.4 % | +13.8 % | +17.4 % | +10.3 % |
| avif (s2) | ≥ s8 | +15.9 % | +17.7 % | +20.7 % | +12.5 % |

So the issue's premise measures out: a budget that caps JXL at e5 costs it +5–14 %
bytes while WebP capped at m4 loses +4–8 % — the family argmin must see these
offsets or it will keep picking the family whose *uncapped* RD is best.

Per-effort cost (`ms/MP` p50, medium size class; fit `ln ms = a + b·ln px` in the TSVs):
webp m2 / m4 / m6 = 95 / 131 / 238; avif s8 / s6 / s4 / s2 = 2161 / 2866 / 6867 / 11759.

**Data caveats (do not calibrate a cost model past these without fixing them):**

- **zenjxl `encode_ms` is not the encoder's wall time.** It is ~10 s for a 0.33 MP
  image and flat across e1…e9 (`vd-e1_zen_def` 10.0 s, `vd-e9_zen_def` 9.7 s at the
  same pixel count). The jxl canonical's timing column carries something else (a
  whole-cell wall clock from the sweep worker, most likely); its degradation table
  is valid, its cost table is not. Re-time jxl per effort before using it for the
  feasibility mask.
- **zenjpeg has no effort axis** in `modes_full` (strategy × trellis × subsampling
  cells; no speed knob), so no degradation table — JPEG's feasible effort is its
  only effort. Its `viable` cost estimate still needs the codec's own model.
- The tables use the validate split for jxl/avif (statistic is per-codec, not
  per-split; train+validate+test for webp). Sizes are ≤ 1 MP (the canonical corpus
  has no `large` renditions) — the `tiny` ms/MP is fixed-cost dominated, read the
  fits.

## What is NOT landed, and the decision it needs

The offsets cannot be threaded into `RouteDecision::resolve` as a plain
`per_output` add, because the shipped lossy router does not emit per-family
scores in any bytes space: it emits six **logistic-regression logits**
(`zenmetrics/scripts/picker/pairwise_discriminants.py`, "A beats B at target")
which `route::pairwise_round_robin` sums into a rank score. A ln-bytes offset
`δ_A − δ_B` has no meaning added to a logit. Two ways to make the degradation
compose, both trainer-side contract changes — **pick one**:

1. **Offset path (issue's "declared metric space").** Re-fit the six pairwise
   heads as **ln-bytes-ratio regressors** (`margin = E[ln B_B − ln B_A]` at the
   target, reference efforts) instead of logits, declare
   `zenpicker.score_space = "ln_bytes"` in the bake, and let the runtime add
   `δ_B(cap_B) − δ_A(cap_A)` to each pair margin before the round-robin (a
   sigmoid of a ln-bytes margin is still a monotone "A wins" belief). Runtime
   change: `RouteDecision::resolve` gains an offsets parameter (or a new
   `resolve_with_degradation`), `MetaPicker::route` takes the per-family
   feasible-effort caps and looks the offsets up from a table shipped in the bake
   metadata (`zenpicker.degradation = <family>:<cap>:<target>:<δ>` rows, this
   session's TSVs are the source). Public-API addition on zenpicker 0.x.
2. **Scorer path (#55/#56 style).** Keep the logits; fit **per-budget-tier**
   discriminants at training time with each family's bytes replaced by
   `bytes_at(feasible effort for that tier)` — one router bake per
   `EncodeMode`/preset (Fastest / Realtime / Offline), selected at load. No new
   runtime arithmetic; the cost is N bakes and the preset → tier mapping
   (zenpipe#28).

Either way the runtime signature grows: `MetaPicker::route(..)` today takes
`per_family_est_ms` (one estimate per family); with degradation it needs the
**feasible effort per family** under the budget (the codec cost models resolve
that from `EncodeBudget` + pixels; zencodecs owns them). That is the public-API
change waiting on sign-off; nothing in this repo should be added speculatively
until (1) vs (2) is chosen, because the two need different bake contracts.

Fitting inputs are in tree: `zentrain/tools/fit_family_degradation.py` (tests in
`test_canonical_tools.py`), the three tables, and `canonical_to_pareto.py` for
any re-fit that needs train_hybrid's shape.
