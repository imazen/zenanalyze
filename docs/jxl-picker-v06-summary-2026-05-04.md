# zenjxl picker v0.6 — followup summary 2026-05-04

Continuation of `jxl-picker-investigation-2026-05-04.md`. The investigation
doc identified the v0.5 HOLD as a teacher-labeling bug. This doc records
what happened when we tried to ship the fix through the existing trainer
pipeline, and what does work.

## TL;DR

**A safety-aware MLP classifier picker on existing v05c data ships with
clean wins**:

| Metric | v0.5 picker | v0.6 safety-MLP regress | v0.6 MLP classifier |
|---|---:|---:|---:|
| Verdict | HOLD | HOLD | **SHIP** |
| Mean Δbytes | +3.11% | +1.18% | **-1.51%** |
| Mean Δzensim | +1.62pp | +0.43pp | +0.15pp |
| Mean Δms | n/a | n/a | **-5.93%** |
| Picker prefers default | 9% | 4% | 68% |
| Pick-departs-default Δbytes | n/a | n/a | -4.73% |

All three distance bands (tight ≤1.0, mid 1..3, loose >3) show wins
under the classifier. The classifier picks effort=7 default 68% of
the time and only departs when a strict alternative is available.

## What was tried (in order)

### 1. Add `--safety-default-cell` to train_hybrid.py

Added to `build_dataset` a per-(image, zq) mask that hides any
alternative cell whose min-bytes config either takes more than
`--safety-speed-tol` (default 1.05) times the default cell's encode
time, OR fails to deliver a `1 - --safety-bytes-min-gain` (default
0.99) bytes savings. Diff lands in `train_hybrid.py:830-870` (mask
logic) + `train_hybrid.py:1980+` (CLI args) + caller wiring.

Trained with `--safety-default-cell effort7`. Headline numbers:

| | v0.5 baseline | v0.6 safety mask |
|---|---:|---:|
| Student val argmin_acc | 51.0% | **78.9%** |
| Train→val gap | +6.0pp | +1.89pp |
| Mean overhead | 6%+ | 2.89% |
| Cell preference | 60% effort3 | 36% effort9, 36% effort3 |

The student MLP fits the safety-masked teacher well — argmin accuracy
jumped 28pp. The mask makes the teacher choice "default unless a
clearly safer alternative" which is a much easier signal to fit.

### 2. Distance-banded A/B harness — first try (HOLD)

Existing `holdout_ab_lookup_jxl.py` queried the picker once per image
with `zq=75` dummy, then evaluated the chosen cell at every distance
in the sweep. Result: HOLD, +1.18% bytes / +0.43pp zensim. Picker
preferred effort=9 36% of the time, regressing bytes at tight/mid
distances for marginal quality wins.

### 3. Fix harness to query at matched zq per distance

For each (image, distance), look up the default cell's actual zensim
at that distance, round to int, and query the picker with that zq.
This matches what a runtime caller would do: they always know the
target quality before invoking the picker.

Result: HOLD, +1.55% bytes / +0.54pp zensim. Slightly worse than
the dummy-zq query because the picker now picks effort=9 48% of the
time for tight-distance high-quality targets, where it produces
larger files than effort=7 for marginal zensim gain.

The picker's regression-then-argmin chain is fragile under the safety
mask: cells whose bytes are masked to ∞ in training don't get a
"don't pick me" signal that the regressor can use at inference. So
the regressor predicts a bytes value for them anyway, and argmin
sometimes picks them.

### 4. MLP CLASSIFIER picker (the win)

Trained a 92→128→128→5 MLP with **softmax classifier head** over the
five effort cells. Input is **`(image features ⊕ log(distance))`** —
matches the prototype `picker_v06_proto.py` shape. Teacher signal is
the same per-(image, distance) safety-constrained label.

Differences from train_hybrid.py's flow:
- Output is N-class softmax, not bytes-log regression.
- Input includes log(distance) directly, not zq via xe-engineering.
- Trained with class-balanced cross-entropy (weights clamped to
  [0.5, 5.0] to avoid blow-up on rare classes).
- Held-out 248-image split, seed 7.

Final val accuracy: 72.4%. A/B verdict: **SHIP**.

| | mean Δbytes | mean Δzensim | mean Δms |
|---|---:|---:|---:|
| All cells | -1.51% | +0.15pp | -5.93% |
| When picker departs from default (32%) | -4.73% | +0.47pp | -18.52% |
| tight (d≤1.0) | -1.55% | +0.10pp | -11.84% |
| mid (1..3) | -1.23% | +0.13pp | +3.37% |
| loose (>3) | -1.68% | +0.21pp | -6.67% |

The structural difference that mattered: **classifier-on-cells
beats regress-bytes-then-argmin under a safety mask**. The classifier
explicitly outputs "this cell is safer," not "predict each cell's
bytes then argmin" which can be fooled by masked-out cells.

## Productionization gap

The MLP classifier prototype isn't yet in a shape the existing baker
pipeline can consume. Three paths:

### A. Hack the existing pipeline (fast, ugly)

Emit the classifier as if its outputs were bytes_log values: for each
softmax logit `l_c`, set bytes_log[c] = `-l_c`. Then the runtime's
`argmin(bytes_log)` becomes `argmax(logit)` — picks the right class.

The scalar head (distance) needs to come from somewhere; the existing
baker emits `distance` per cell. Could either:
- Hardcode distance per cell (constant per cell — picker only chooses
  cell, runtime caller passes its own distance).
- Train a separate small regressor for the scalar head.

**Risk**: this overloads `bytes_log` semantically. Future readers of
the manifest will see a model that "predicts bytes" but it's actually
log-probabilities. Would mark in metadata, but it's still confusing.

### B. Add classifier mode to train_hybrid.py + bake_picker.py

Cleaner long-term. Trainer flag `--head-mode classifier` that emits a
softmax-logit head; baker reads the manifest and writes the model with
a `head_kind: "classifier"` flag the runtime checks. Runtime adds a
small branch: classifier → argmax instead of argmin.

Estimate: ~200 lines across trainer + baker + zenjxl runtime. One
day's work.

### C. Build the classifier as a sidecar picker

Don't touch the existing baker. Write a new `bake_classifier.py` that
emits a separate ZNPR binary with a different schema_hash. zenjxl
runtime gets a new entry point that loads classifier-mode pickers
explicitly.

Estimate: ~400 lines. More code but fully clean separation.

**Recommendation**: B. Smallest production-side change for a feature
we'll want for other codecs too (zenavif's picker also overpredicts
"high quality" cells in its loose band).

## What about the expert JXL knobs?

The expanded `butteraugli_iters / zensim_iters / pixel_domain_loss` etc.
shipped in `turbo-metrics` PR #2 are still valuable for a SEPARATE
"max quality" picker tier. Local validation found:

- Speed-safe regime (≤5% slower than default): 43% wins, mean -3.32%
  bytes, +0.40pp zensim, -61% encode time. **All top winners are
  `butteraugli_iters=0`** — the iters knobs don't help in this regime.
- Speed-unconstrained regime: 67% wins, mean -3.97% bytes, +0.29pp
  zensim, +103% encode time. Top winners DO include `biters=1` and
  `biters=2` cells.

So the expert knobs justify a v06 sweep IF we want a "highest quality
allowed" picker tier on top of the speed-safe v0.6. Cost estimate
~$5-10 for a 300-image clustered subset. Not blocking the speed-safe
ship; should follow it.

## Bug filed: zenjxl-decoder issue #15

Sweep also caught a decoder bug: `effort=9 + distance ≤ 0.5 + screen
content` produces files that jxl-oxide accepts but zenjxl-decoder
rejects with `Invalid AC: N nonzeros after decoding block`. Filed at
imazen/zenjxl-decoder#15 with a clean repro and code-path analysis.

## Artifacts

- v0.5 picker (SHA-locked baseline): `benchmarks/zenjxl_picker_v0.5_2026-05-04.bin{,.bak}` + manifest
- v0.6 safety-mask MLP regressor (HOLD verdict): `benchmarks/zenjxl_hybrid_v06_safety_2026-05-04.json`, baked to `zenjxl_picker_v0.6_safety_2026-05-04.bin`
- v0.6 MLP classifier prototype (SHIP verdict, not yet baked): `/home/lilith/sweep-data/picker_v06_mlp.py`
- A/B reports under `benchmarks/zenjxl_picker_v0.6_safety_holdout_ab*.md`
- Investigation chain: `docs/jxl-picker-investigation-2026-05-04.md`
- v06 sweep grid (turbo-metrics): `scripts/sweep/generate_jobspecs_v06.py`

## Recommended next moves

1. **Decide: A, B, or C above** for productionizing the classifier. Then
   ship the v0.6 picker artifact for review.
2. **Run small v06 sweep** with butteraugli_iters added on a 300-image
   clustered subset (~$5) to test if a separate max-quality picker is
   worth shipping alongside.
3. **Address zenjxl-decoder#15** so the sweep stops dropping cells.
