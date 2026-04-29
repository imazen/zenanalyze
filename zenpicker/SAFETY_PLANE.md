# Safety plane for the zenjpeg Zq picker — design

Defends against the picker's worst-case failure mode: an out-of-distribution or adversarial image where the MLP confidently picks a config that produces zensim *catastrophically below* the user's `target_zq`. A 5–15% byte overage is acceptable; a 30-point quality miss is a product failure.

## Existing scaffolding (what we already have)

- `zenjpeg/src/encode/zq.rs` defines a closed-loop `ZqTarget { max_passes, max_undershoot, BlockArtifactBound }` and a public `EncodeMetrics { achieved_score, achieved_max_block_artifact, passes_used, bytes, targets_met }`. `BytesEncoder::finish_with_metrics()` returns it (`encode/byte_encoders.rs:543`).
- `zenpicker` is codec-agnostic — argmin index over an `n_outputs` mask, no encode/decode/zensim coupling. API frozen at 0.1.x; only additive changes.

The safety plane is **codec-side**, layered on top of the iteration loop that already exists. We're tightening the loop's failure semantics, baking the picker as the pass-0 starting policy, and exposing the rescue outcome to callers.

## Architecture

```
   ┌──────────────────────────────────────────────────────────────┐
   │  Caller (imageflow, CLI)                                     │
   │  asks for Quality::Zq(85) on user-uploaded bytes             │
   └────────────────────┬─────────────────────────────────────────┘
                        │ EncodeRequest, source pixels
                        ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  zenjpeg encoder (controller)                                  │
 │                                                                │
 │  ┌──────────────┐    pass 0 (picker pick)                      │
 │  │ zenpicker    │───► EncoderConfig ─► encode ─► JPEG_0        │
 │  │ argmin_masked│                                              │
 │  └──────────────┘                                              │
 │                            │                                   │
 │                            ▼                                   │
 │                  ┌──────────────────┐                          │
 │                  │ verifier (zensim)│  cheap pre-filter +      │
 │                  │ decode + diffmap │  feature-flagged         │
 │                  └────────┬─────────┘                          │
 │                           │ achieved_zq                        │
 │                           ▼                                    │
 │      target met (achieved_zq ≥ target − rescue_threshold)?     │
 │            │ yes                          │ no                 │
 │            ▼                              ▼                    │
 │       ship JPEG_0                 pass 1 (rescue)              │
 │         + metrics                 conservative-bump config     │
 │                                   ──► encode ─► JPEG_1         │
 │                                          │                     │
 │                                          ▼                     │
 │                              verify (mandatory)                │
 │                                  │                             │
 │                                  ▼                             │
 │                          ship best-of (JPEG_0, JPEG_1)         │
 │                          + picker_rescued=true                 │
 │                          + warn on residual undershoot         │
 └────────────────────────────────────────────────────────────────┘
```

**Two-shot maximum.** No third attempt under any flag. Adversarial input can't trigger an unbounded compute path through this controller — bounded at exactly 2× happy-path cost.

## Verify / rescue protocol

**Triggers and constants** (live as `pub` fields on a new `RescuePolicy` struct under `ZqTarget`):

- `verify: VerifyMode = OnUndershootRisk` — `Always | OnUndershootRisk | Never`.
- `rescue_threshold_pp: f32 = 3.0` — triggers when `achieved_zq < target_zq − rescue_threshold_pp`. **Initial value is a placeholder; calibration is item 6 below.**
- `rescue_strategy: RescueStrategy = ConservativeBump` — `ConservativeBump | SecondBestPick | KnownGoodFallback`.

**Step-by-step:**

1. **Pick.** `Picker::argmin_masked(features, mask)` → cell index → `EncoderConfig`. Cache the second-best index (single extra pass over model output, same `predict` call).
2. **Encode pass 0** → `JPEG_0`, `bytes_0`.
3. **Cheap pre-filter** (default on). Run verification only when:
   - `target_zq ≥ 85` (high-quality SLA band where misses hurt most), OR
   - the picker's predicted bytes for the chosen cell were in the bottom decile of the cell's training distribution (signal: model is far from training data).
   Otherwise skip — happy path stays at ~60ms.
4. **Verify.** Decode `JPEG_0`, compute zensim against source. Cost: ~3ms decode + ~1-2ms zensim at 1MP, ~3ms + ~6-8ms at 4K.
5. **Decision:**
   - `achieved_zq ≥ target_zq − rescue_threshold_pp` → ship `JPEG_0`, `passes_used=1`, `picker_rescued=false`.
   - Else → rescue (pass 1).
6. **Rescue pick** (`ConservativeBump` default):
   - Bump quality by `max(target_zq − achieved_zq + safety_margin, +5)` jpegli-q points, **and** force `Subsampling::S444` (no chroma loss), **and** disable XYB if it was on. Single deterministic transform.
   - Alternative `SecondBestPick`: use cached 2nd-place argmin, but only if its predicted bytes are ≥1.15× the first pick. Falls through to `ConservativeBump` if too close.
   - `KnownGoodFallback`: hardcoded `EncoderConfig::ycbcr(min(target_zq + 7, 95), Subsampling::S444).baseline()` — empirically never undershoots by more than ~1pp on the held-out corpus.
7. **Encode pass 1** → `JPEG_1`, verify (mandatory — we promised a quality floor).
8. **Two-shot terminate.** Ship `argmax(JPEG_0.zq, JPEG_1.zq)`. If both undershoot, ship the **higher zq** (NOT smaller bytes — quality-miss is the failure mode). Set `targets_met=false`, `picker_rescued=true`, attach `warning=SafetyMissBeyondRescue`.

## zenjpeg API surface (additive, all on existing `#[non_exhaustive]` types)

```rust
pub struct ZqTarget {
    // ... existing fields ...
    pub rescue: Option<RescuePolicy>,  // None = current behavior, no safety plane
}

#[non_exhaustive]
pub struct RescuePolicy {
    pub mode: VerifyMode,
    pub rescue_threshold_pp: f32,
    pub strategy: RescueStrategy,
}

pub struct EncodeMetrics {
    // ... existing fields ...
    pub picker_first_pick_zensim: f32,             // NaN if not verified
    pub picker_rescued: bool,
    pub picker_rescue_strategy_used: Option<RescueStrategy>,
    pub picker_warning: Option<PickerWarning>,     // SafetyMissBeyondRescue | VerifySkipped | …
}
```

Imageflow scrapes these per request to detect attack waves: sustained `picker_rescued=true` rate spike or any `picker_warning=SafetyMissBeyondRescue`.

## zenpicker API surface (additive only)

All shipped in 0.1.x. No schema change, no `.bin` format change.

```rust
// Cached second-best for SecondBestPick rescue.
impl<'a> Picker<'a> {
    pub fn argmin_masked_top_k<const K: usize>(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        adjust: Option<CostAdjust<'_>>,
    ) -> Result<[Option<usize>; K], PickerError>;

    pub fn argmin_masked_top_k_in_range<const K: usize>(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        adjust: Option<CostAdjust<'_>>,
    ) -> Result<[Option<usize>; K], PickerError>;
}

// Codec-agnostic decision logic.
pub mod rescue {
    pub struct RescuePolicy { pub rescue_threshold_pp: f32, pub strategy: RescueStrategy }
    pub enum RescueStrategy { ConservativeBump, SecondBestPick, KnownGoodFallback }
    pub enum RescueDecision { Ship, Rescue }
    pub fn should_rescue(achieved_zq: f32, target_zq: f32, &RescuePolicy) -> RescueDecision;
}

// Reach gate as a runtime parameter — codec / caller picks threshold.
pub fn reach_gate_mask(reach_rates: &[f32], threshold: f32, out: &mut [bool]);
```

`K=2` covers the second-best-pick use case. The reach gate is a *parameter*, not a baked constant — the bake records raw `reach_rate[c]` per `target_zq` in the manifest's `reach_safety` table, and codec consumers convert with `reach_gate_mask` at request time using whatever threshold matches the caller's `QualityIntent` (0.99 strict / 0.95 high-q / 0.0 max-quality).

The bake-time tripwire ("predicted-bytes confidence quantile per cell" used by the cheap pre-filter) is bake-side metadata: pack a `[f32; n_cells]` of the bottom-decile threshold into the `.bin`'s post-header extension area (the format already advertises `header_size` for forward compat). Picker exposes it via a passthrough getter — *not yet implemented*; `reach_safety` covers the SLA-mode gate today.

## Adversarial-corpus regression test

**Where it lives:** `zenanalyze/zenpicker/testdata/adversarial/` (the picker is the thing that fails, so the corpus belongs near the picker). Codec borrows it at test time via dev-dependency on a small fixture-loader crate.

**What it asserts:** for each `(image, target_zq)` pair, the safety-plane-enabled zenjpeg encode must produce `achieved_zq ≥ target_zq − 5.0` — a **hard floor** wider than the rescue threshold. Failure here means rescue itself didn't save us. Soft assertion: `picker_rescued` rate ≤ 25% on the corpus (otherwise picker has drifted; retrain rather than just rely on rescue).

**How new images are added:** a CLI tool `zq-tripwire add-image <png> --target-zq 85` runs the encoder *without* rescue, and if `achieved_zq < target_zq − 1.0` it appends to the manifest with the observed first-pick achieved score recorded. Corpus grows as failures are discovered in production (imageflow logs surface them via `picker_warning`).

## Open questions / data needed before freeze

1. **Threshold calibration.** Need the held-out **p99 zensim shortfall** across the 1388-image training set to set `rescue_threshold_pp` defensibly. Worst single-target hit was +1.65pp byte overhead; the equivalent quality-shortfall stat tells us where the natural threshold is.
2. **Pre-filter ROC.** Does `picker_predicted_bytes < bottom_decile_for_cell` actually correlate with shortfall, or is it noise? Without correlation the cheap filter is dead weight; just always-verify when `target_zq ≥ 80`.
3. **Cost of always-verify.** Always-verify is ~12% overhead at 4K, ~7% at 1MP. **If those numbers hold, default to `VerifyMode::Always` for `target_zq ≥ 85` and skip the pre-filter complexity entirely.**
4. **Does `ConservativeBump` actually rescue?** Need a small targeted study: for the 16 worst-undershoot held-out images, does +5 jpegli-q + force-444 always reach target? If even one still misses, rescue strategy needs revising.
5. **Should rescue cost feedback-train the picker?** Every `picker_rescued=true` event is labeled training data for free. zenanalyze logs could be the corpus for v2 picker. Out of scope for this PR but worth noting.

## Critical files for implementation

- `zenjpeg/zenjpeg/src/encode/zq.rs` — the iteration loop; gains `RescuePolicy` field
- `zenjpeg/zenjpeg/src/encode/byte_encoders.rs` — `EncodeMetrics` extension
- `zenjpeg/zenjpeg/src/encode/encoder_types.rs` — `RescueStrategy` / `PickerWarning` enums
- `zenanalyze/zenpicker/src/lib.rs` — `argmin_masked_top_k`, `reach_gate_mask` re-exports
- `zenanalyze/zenpicker/src/mask.rs` — `argmin_masked_top_k`, `reach_gate_mask` impl
- `zenanalyze/zenpicker/src/rescue.rs` — `should_rescue`, `RescuePolicy`, `RescueStrategy`, `RescueDecision`
- `zenanalyze/zenpicker/tools/train_hybrid.py` — `--objective {size_optimal,zensim_strict}`, `--reach-threshold`
- `zenanalyze/tools/bake_picker.py` — `safety_profile` / `training_objective` / `reach_safety` manifest passthrough
