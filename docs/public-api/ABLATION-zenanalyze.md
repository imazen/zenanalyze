# Public-API Ablation Report: zenanalyze

**Date:** 2026-06-11  
**Snapshot commit:** 52c43de (main@origin)  
**Crate version:** 0.2.0  
**Snapshot items (default/all-features):** 260 / 337  
**Governance:** 0.2.x — breaking changes permitted; bump minor for breaks, patch for additive. See `CLAUDE.md`.

**Grep template (as of this scan):**
```
grep -r --include="*.rs" --include="*.py" "SYMBOL" /home/lilith/work/zen/ \
  2>/dev/null | grep -v "/zenanalyze/" | grep -v "/target/" | grep -v "/.jj/"
```

---

## Summary

| Class | Count | % of 260 |
|-------|------:|--------:|
| A — doc(hidden) / deprecated candidate | 1 | 0.4% |
| B — pub(crate)/remove, queued breaking | 0 | 0% |
| **Total flagged** | **1** | **0.4%** |

---

## Surface overview

The 260-item default surface is almost entirely the `feature` module:
`AnalysisFeature` (102 variants, stable ID contract), `FeatureSet`, `AnalysisQuery`,
`AnalysisResults`, `FeatureValue`, `ImageGeometry`, `PackError`, `MissingFeatures`.
Root-level: `analyze_features`, `analyze_features_rgb8`, `try_analyze_features_rgb8`,
`AnalyzeError`. The 77 extra items in all-features are `experimental` + `hdr` gated
feature variants — all intentional.

External consumers confirmed (grep): jxl-encoder (`analyze_features_rgb8`, `AnalysisFeature`,
`AnalysisQuery`, `FeatureSet`, `AnalysisResults`, `FeatureValue`), zenmetrics/zen-cloud-vastai
(`AnalysisQuery`, `FeatureSet`). PackError + MissingFeatures used by zenjpeg picker.

---

## Flagged items

### A-class: `#[doc(hidden)]` candidate

| Item | Module | External hits | Rationale |
|------|--------|:-------------:|-----------|
| `pub fn __analyze_internal` | `zenanalyze` (root) | 0 (zero org-wide) | Double-underscore prefix signals "not for callers." Parameter type `feature::InternalQuery` has all-`pub(crate)` fields so the function is unreachable from outside the crate in practice — cargo-public-api omits it from the snapshot because callers cannot construct the argument. However the `pub` visibility is misleading. Adding `#[doc(hidden)]` makes the intent explicit and matches convention without requiring a breaking change. |

**Governance cost:** Additive (`#[doc(hidden)]` is not a breaking change — no callers exist). Zero minor-bump needed.

---

## Not flagged (with rationale)

- **`AnalysisFeature` 102-variant enum** — the product's stability contract; IDs are immutable. KEEP wholesale.
- **`PackError`, `MissingFeatures`** — used by zenjpeg picker as error variants. KEEP.
- **`analyze_features` (the `PixelSlice` entry point)** — primary API; multiple external callers. KEEP.
- **`try_analyze_features_rgb8`** — parallel fallible entry; migration artifact from 0.1→0.2 but still a public promise. KEEP.
- **`experimental` and `hdr` gated variants** — intentional opt-in surface. KEEP.
- **`AnalyzeError`** — required as the `Result` error type. KEEP.
- **`InternalQuery` struct** — nominally `pub` but all fields are `pub(crate)`; callers cannot construct it; effectively sealed. Not a mistake.
