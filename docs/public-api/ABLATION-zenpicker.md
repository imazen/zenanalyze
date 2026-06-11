# Public-API Ablation Report: zenpicker

**Date:** 2026-06-11  
**Snapshot commit:** 52c43de (main@origin)  
**Crate version:** 0.1.0  
**Snapshot items (default/all-features):** 94 / 94  
**Governance:** 0.1.x.

**Grep template (as of this scan):**
```
grep -r --include="*.rs" --include="*.py" "SYMBOL" /home/lilith/work/zen/ \
  2>/dev/null | grep -v "/zenanalyze/" | grep -v "/target/" | grep -v "/.jj/"
```

---

## Summary

| Class | Count | % of 94 |
|-------|------:|-------:|
| A — doc(hidden) / deprecated candidate | 0 | 0% |
| B — pub(crate)/remove, queued breaking | 0 | 0% |
| **Total flagged** | **0** | **0%** |

---

## Surface overview

The 94 items decompose as:

- `CodecFamily` enum (6 variants: Jpeg=0, Webp=1, Jxl=2, Avif=3, Png=4, Gif=5) + trait impls
- `CodecFamily::ALL`, `CodecFamily::COUNT`, `CodecFamily::index`, `CodecFamily::label`
- `AllowedFamilies` struct + `all`, `none`, `allow`, `deny`, `is_allowed`, `from_allowed`, `as_slice`, `any`
- `MetaPicker<'b>` struct + `new`, `pick`, `predictor`, `validate_family_order`
- `MetaPickerError` enum + Display/Debug/Error impls
- `ALL_LABELS_CSV` const, `FAMILY_ORDER_KEY` const

The 94 items are the same for default and all-features (no feature flags).

External consumers confirmed: zenpicker is the cross-codec router consumed by orchestrators
and eventually by the pipeline layer. `CodecFamily`, `AllowedFamilies`, and `MetaPicker` are
the entire intended surface.

---

## Not flagged items (with rationale)

- **`CodecFamily`** — Stable enum with discriminants that match baked model output indices.
  Variants are the product contract. KEEP.
- **`CodecFamily::ALL`, `COUNT`, `index`, `label`** — Utility surface used for iteration
  and CSV label generation. KEEP.
- **`AllowedFamilies`** — The filter mask for `MetaPicker::pick`. KEEP.
- **`MetaPicker::predictor()`** — Returns `&mut Predictor` for callers that need to drive
  the predictor directly (advanced use case). KEEP.
- **`MetaPicker::validate_family_order()`** — Sanity gate for integration tests. KEEP.
- **`ALL_LABELS_CSV`, `FAMILY_ORDER_KEY`** — Protocol constants for bake-time and
  runtime handshake. KEEP.
- **`MetaPickerError::FamilyOrderMismatch` fields** — The struct variant fields (`actual`,
  `expected`) are pub, which allows callers to format the mismatch. Intentional. KEEP.
