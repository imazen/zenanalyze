# Public-API Ablation Report: zenpredict

**Date:** 2026-06-11  
**Snapshot commit:** 52c43de (main@origin)  
**Crate version:** 0.2.0  
**Snapshot items (default/all-features):** 747 / 1027  
**Governance:** 0.2.x — breaking changes permitted; bump minor for breaks, patch for additive.

**Grep template (as of this scan):**
```
grep -r --include="*.rs" --include="*.py" "SYMBOL" /home/lilith/work/zen/ \
  2>/dev/null | grep -v "/zenanalyze/" | grep -v "/target/" | grep -v "/.jj/"
```

---

## Summary

| Class | Count | % of 747 |
|-------|------:|--------:|
| A — doc(hidden) / deprecated candidate | 1 | 0.1% |
| B — pub(crate)/remove, queued breaking | 0 | 0% |
| **Total flagged** | **1** | **0.1%** |

---

## Surface overview by module

| Module | Items (default) | External callers |
|--------|:--------------:|-----------------|
| `argmin` | 85 | zenjpeg, zenwebp, zenavif, zensim (`argmin_masked_in_range`, `AllowedMask`, `ScoreTransform`, `threshold_mask`) |
| `output_spec` | 84 | zensim training (`OutputSpec`), zenpredict-bake JSON pipeline |
| `Model` (model.rs) | 52 | All codec pickers (`Model::from_bytes`), zensim MLP runtime |
| `PredictError` | 44 | zenjpeg, zenwebp picker error chains |
| `wire` | 36 | zenpredict-bake composer only (legitimate cross-crate need — both sides must use identical offsets) |
| `keys` | 34 | zensim training (`FEATURE_TRANSFORMS`, `FEATURE_TRANSFORM_PARAMS`) |
| `rescue` | 26 | `advanced` feature-gated; zenjpeg `__picker-research` feature only |
| `Section` | 24 | zenpredict-bake composer (wire format serialization) |
| `FeatureTransform` | 22 | zensim training, codec pickers |
| `Metadata` | 20 | zenpredict-bake JSON pipeline |
| `FeatureBound` | 17 | zenjpeg picker (`first_out_of_distribution`) |
| `Predictor` | 12 | All codec pickers and zensim MLP runtime |
| `limits` | 8 | zenpredict-bake validation only (cross-crate) |
| `CellHint`, `SafetyCompact` | 8 | `advanced`-gated; not imported outside zenanalyze workspace |
| `Activation`, `WeightDtype` | 8 each | zensim training, zenpredict-bake JSON pipeline |

The 747→1027 gap (280 extra items in all-features) is entirely the `advanced` feature, which gates `rescue`, `safety`, extended `argmin` overloads, and `OutputValue`/`apply_spec`. The `advanced` feature is an explicitly internal research gate.

---

## Flagged items

### A-class: `#[doc(hidden)]` candidate

| Item | Module | External hits | Rationale |
|------|--------|:-------------:|-----------|
| `pub const LEAKY_RELU_ALPHA: f32` | `model.rs` (re-exported at crate root) | 0 org-wide | No external crate imports this constant. zensim training code that needs the value hardcodes `0.01` in a comment "matches the runtime's hardcoded LEAKY_RELU_ALPHA" — they deliberately do NOT import it to avoid the training dep taking on the runtime crate. Hiding the constant does not help callers; it only prevents accidental coupling. `#[doc(hidden)]` makes the intent explicit. Breaking only if someone wrote `use zenpredict::LEAKY_RELU_ALPHA`, which grep finds zero instances of. |

**Governance cost:** Technically a breaking change (removing from public docs), but zero callers means zero real cost. Could be done as B-class (make `pub(crate)`) in a subsequent minor bump; A-class (`#[doc(hidden)]`) is non-breaking and achievable now.

---

## Not flagged (with rationale)

- **`wire` module (36 items)** — Used exclusively by zenpredict-bake composer, but that is the documented cross-crate need. Both parser and composer MUST share the same byte offsets or they drift silently. Keeping it pub is correct; the doc comment on `wire.rs` explains this explicitly.
- **`FORMAT_VERSION`** — Used by zenpredict-bake composer (`buf[4..6] = FORMAT_VERSION.to_le_bytes()`). KEEP.
- **`Header`, `LayerEntry`, `Section`** — Used by zenpredict-bake composer for wire format construction. Cross-crate need. KEEP.
- **`LayerView`, `WeightStorage`** — Used by zensim validate tools (`bake_quant_stats`, `inspect_l0_input_norms`, `monotone_cbc_projection`) and zenjpeg picker research. KEEP.
- **`LayerIter`** — Returned by `Model::layers()`, used by zensim validate tooling. KEEP.
- **`limits` module** — `MAX_DIM` used by zenpredict-bake for input validation. Cross-crate need. KEEP.
- **`keys` module** — `FEATURE_TRANSFORMS`, `FEATURE_TRANSFORM_PARAMS` used by zensim training pipeline. KEEP.
- **`rescue` + `safety` (advanced feature)** — Explicitly gated; used by zenjpeg `__picker-research` internal feature. Not a public stable surface. KEEP as-is.
- **`argmin_masked_in_range`, `AllowedMask`, `ArgminOffsets`** — Multiple external codec callers. KEEP.
- **`first_out_of_distribution`** — Used by zenjpeg picker OOD gate. KEEP.
- **`BakeRequestJson` / `bake_from_json`** — Canonical Python→Rust API for the zentrain bake pipeline. KEEP.
- **`apply_zero_bias*` functions** — Used by zensim training pipeline. KEEP.
