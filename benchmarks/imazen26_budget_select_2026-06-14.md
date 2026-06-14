# imazen-26 budget-first training-corpus selection (2026-06-14)

Builds a **diverse training corpus under a gigapixel budget** (not a fixed K):
thumbnail every eligible image (near-free coverage floor), then spend the
remaining pixel budget on the most feature-distinct resize renditions via
farthest-point sampling (FPS). FPS is incremental, so the manifest is a
**priority-ordered list — any GP budget is a prefix** (`cumulative_gp`).

Supersedes the K-means ablation (`imazen26_cluster_ablation_2026-06-14.md`):
budget-first directly answers "how much diversity fits in B GP", and resizes are
prioritized over crops (resizes matter more for picker training).

## Hold-out split — EVEN leading id = train, ODD = hold-out

Train-eligible = images whose leading filename number is **even** (`5300` yes,
`5301` no). Clean ~50/50, balanced within every content class: **1082 train /
1075 hold-out**, 0 unnumbered. Variants preserve the leading id
(`5302.scale512x384`), so the split propagates to every derivative — no leakage.

## Coverage vs gigapixel budget

p95/p50 = distance from the *whole* even-id resize pool (14,211 candidates,
1082 images) to the selected prefix; lower = better coverage.

| budget | variants | p50 | p95 | max |
|--:|--:|--:|--:|--:|
| 0.25 GP | 1,107 | 5.46 | 11.0 | 18.9 |
| 0.5 GP | 1,145 | 5.44 | 10.1 | 13.5 |
| **1.5 GP** | **1,482** | **4.83** | **6.73** | **7.58** |
| 1.0 GP | 1,300 | 5.15 | 7.63 | 9.04 |
| 2.0 GP | 1,759 | 4.41 | 5.90 | 6.62 |

Thumbnail floor (every image @ ≤128px) costs only **12.7 MP** — essentially
free, as predicted. Biggest coverage gain is 0.5→1.0 GP; steady after.

## Recommendation: 1.5 GP (≈1,482 renditions)

Every even image covered, all 21 classes present, good coverage (p95 6.73),
size mix = 1,260 thumbnails + ~400 larger (88 of them >2048px — FPS favors
full-detail renditions as most feature-distinct). Encode-sweep cost ≈
`1.5 GP × n_q × n_configs` (e.g. 25 q × 12 configs ≈ 450 GP — fleet-feasible).
Trivially retunable: truncate the manifest at any `cumulative_gp` (1.0 = leaner,
2.0 = richer).

## Variant naming (op-chain, id-preserving)

`<even-id>.scale<W>x<H>[.crop<x>.<y>.<w>.<h>]…` — chainable scale→crop; crops as
whole-% or absolute (a later pass). This run emits resize variants only, e.g.
`1000.scale4032x3024`, `1000.scale128x96`.

## Artifacts (block storage + Tower; not in git)

- Ordered manifest: `/mnt/v/output/imazen-26-features/imazen26_train_variants_2026-06-14.tsv`
  (1,759 rows to 2 GP; cols `rank, cumulative_gp, image_path, scale_w, scale_h, megapixels, content_class, variant_name`). Tower-mirrored.
- Selector: `benchmarks/imazen26_budget_select_2026-06-14.py`.

## Next passes (not yet done)

1. **Render** the selected variants via **zenresize Mitchell-sharp** (`Filter::Mitchell`
   + sharpen), written as `<variant_name>.png` — the actual training corpus.
2. **Crops**: a lower-priority pass adding cross-image, absolute (e.g. 256×256)
   and scale-then-crop regions (needs a crop-candidate feature extraction).
3. **Hold-out**: odd-id images (1075) for picker evaluation — same renditioning
   when an eval set is needed.
