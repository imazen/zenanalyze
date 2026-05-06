# Picker pipeline schema mismatch — 2026-05-06

## Summary

The current pipeline of (content_classifier_v0.2 → meta-picker_v0.3 →
zenjxl_picker_v0.7b) has **two schema mismatches** that block direct
runtime composition. They are recoverable but require either a
re-bake or a runtime adapter layer; documenting here so the next
implementation session resolves them deliberately.

## Mismatch #1 — content classifier outputs ≠ meta-picker inputs

**content_classifier_v0.2_2026-05-06.bin** (`schema_hash 0x10429ad…`):

- `n_outputs = 4`: photo, screen, lineart, document (no synthetic)
- Trained on rebalanced 4-class corpus

**zenpicker_meta_v0.3_2026-05-06.bin** (`schema_hash 0xd900d519…`):

- `n_inputs = 20` = 14 zenanalyze features + **5** cclass one-hots
  + target_band
- Expects: `cclass_photo, cclass_screen, cclass_lineart,
  cclass_document, cclass_synthetic`

At runtime, the classifier never sets `cclass_synthetic = 1`. Synthetic
content always maps to whichever of the 4 known classes the classifier
chose. This is bias the meta-picker absorbed during training and will
underperform on synthetic content in production.

### Resolution options

**A. Retrain content classifier with 5 outputs** (preferred). Add
synthetic class to the training labels (already exists in the
rebalanced corpus's `gen-synthetic__*` paths). Keeps meta-picker as-is.
Cost: a few minutes to retrain + re-bake.

**B. Re-bake meta-picker with 4 cclass inputs**. Drop `cclass_synthetic`
from inputs, retrain. Loses the training signal already absorbed for
synthetic class. Cost: re-run v12_metapicker_train.py with
`--cclass-cols=4`.

**C. Runtime adapter** that maps classifier 4-output → meta-picker
5-input by setting `cclass_synthetic = 0` always. Already what would
happen by default; just acknowledges the bias. Cheapest, but lowest
quality.

## Mismatch #2 — meta-picker output ≠ MetaPicker enum order

**zenpicker_meta_v0.3** outputs are 3-codec (zenwebp=0, zenjxl=1,
zenavif=2). **`zenpicker::CodecFamily`** has 6 variants
(Jpeg=0, Webp=1, Jxl=2, Avif=3, Png=4, Gif=5).

`MetaPicker::validate_family_order()` checks `family_order` metadata
against `CodecFamily::ALL` and rejects on mismatch. Loading v0.3
directly into `MetaPicker::new()` will fail at validation.

### Resolution options

**A. Bake a 6-family meta-picker** that includes JPEG and PNG slots.
The training data exists for 3 codecs (zenjxl/avif/webp from v10/v12);
v8/v9/v11 partials cover other codecs. Need: sweep zenjpeg + zenpng
on v10 corpus (task #39), join into a 5-codec dataset, retrain. PNG/GIF
columns can stay as -inf-masked outputs for unsupported families.

**B. Loosen `validate_family_order` to accept a subset**. Allow the
runtime `AllowedFamilies` mask to be a subset of `family_order`. The
3-codec bake would declare `family_order = "webp,jxl,avif"` and the
runtime would mask the 3 missing slots with -inf. Requires zenpicker
API change (additive; existing 6-family bakes still work).

**C. Bypass `MetaPicker` and use `zenpredict::Predictor` directly**.
The 3-codec v0.3 bake works as a plain Predictor with `n_outputs=3`.
Caller-side mapping from index 0/1/2 → CodecFamily::Webp/Jxl/Avif.
Cheapest but skips the family-validation safety net.

## Recommended path

1. Sweep zenjpeg + zenpng on v10 corpus (task #39) — gives 5-codec
   coverage.
2. Retrain content classifier with 5 outputs (synthetic + photo +
   screen + lineart + document).
3. Re-bake meta-picker with 5-codec output (jpeg + webp + jxl + avif
   + png) and 5-cclass input. The schema_hash will rotate; old bake
   is retained for reproducibility.
4. Wire 5-codec meta-picker into `zenpicker::MetaPicker` and the
   5-output classifier into `zenanalyze` content-class API.

End state: production pipeline matches the diagram in the autonomous-run
progress doc, with no bias-by-default on synthetic content and no
3-vs-6 family mismatch.

## Production gain implications

The honest oracle ceiling on the FULL holdout is **−13.26%** vs
always-zenjxl (see `tools/honest_oracle_eval_2026-05-06.py`). At
~67% holdout accuracy, the meta-picker will capture **~−7 to −10%
bytes** on mixed traffic — *not* the −16.92% headline that comes
from a multi-codec-filtered subset. The 5-codec rebake won't move
that ceiling much (PNG/JPEG dominate at very-low or very-high
quality where the current band-based eval doesn't reach), so the
priority is correctness, not Pareto chasing.
