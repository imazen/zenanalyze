# Public-API Ablation Report: zenpredict-bake

**Date:** 2026-06-11  
**Snapshot commit:** 52c43de (main@origin)  
**Crate version:** 0.1.0  
**Snapshot items (default/all-features):** 780 / 780  
**Governance:** 0.1.x currently; CLI flags not part of semver surface per `CLAUDE.md`.

**Grep template (as of this scan):**
```
grep -r --include="*.rs" --include="*.py" "SYMBOL" /home/lilith/work/zen/ \
  2>/dev/null | grep -v "/zenanalyze/" | grep -v "/target/" | grep -v "/.jj/"
```

---

## Summary

| Class | Count | % of 780 |
|-------|------:|--------:|
| A — doc(hidden) / deprecated candidate | 0 | 0% |
| B — pub(crate)/remove, queued breaking | 0 | 0% |
| **Total flagged** | **0** | **0%** |

---

## Surface overview by module

| Module | Items | External callers |
|--------|:-----:|-----------------|
| `json` | 164 | `bake_from_json_str` used by zensim training scripts; `BakeRequestJson` is the canonical Python→Rust bake interface |
| `composer` | 86 | `BakeLayer`, `BakeRequest`, `BakeMetadataEntry`, `bake` — used by zensim training pipeline, zenpredict round-trip tests |
| `cli` | 8 | `run_bake_cli`, `run_inspect_cli`, `run_repack_cli` — used by crate's own binary targets only; design is correct |
| `zero_bias` | 10 | `apply_zero_bias_per_layer_in_place` used by zensim training (`mlp_train/mod.rs`) |
| `optimize` | 10 | `bake_optimized` — not confirmed external; internal optimization utility |

The 780 items are the same for default and all-features (no feature flags in this crate).
The 780 count includes all the trait impls, `From` conversions between JSON and runtime types,
and standard derives — these inflate the item count but are all legitimate boilerplate.

---

## Not flagged items (with rationale)

- **`cli` module** — `pub` so binary targets in the same crate can call these functions.
  Per `CLAUDE.md`: CLI binary surfaces are explicitly not part of the semver contract. KEEP.
- **`BakeRequestJson` + `bake_from_json_str`** — The canonical Python→Rust bake API.
  The `tools/bake_picker.py` shell invokes the `zenpredict-bake` CLI (which calls `run_bake_cli`
  → `bake_from_json_str`); zensim scripts invoke it directly via the Rust API. KEEP.
- **`BakeLayer`, `BakeRequest`, `BakeMetadataEntry`, `bake`** — Used by zensim validate
  tools and zenpredict integration tests. KEEP.
- **`apply_zero_bias_per_layer_in_place`** — Used by zensim training. KEEP.
- **`ActivationJson`, `DtypeJson`, `FeatureBoundJson`, etc.** — JSON serde types for the
  Python→Rust bake contract. KEEP: every field maps to a wire concept the Python trainer emits.
- **`From<*Json>` conversions for runtime types** — 4 items; needed for `bake_from_json` to
  convert deserialized JSON into `zenpredict` runtime types. KEEP.
- **`bake_optimized`** — `optimize` module; no confirmed external callers but it is the
  natural next step after `bake` for production-grade bakes. Conservative default: KEEP.
  (If no caller appears in 2 minor versions, revisit as B-class.)
