# Python IQA-stat → zenstats migration plan

**Date:** 2026-05-26.
**Driver:** route every inline reimpl of Mohammadi-2025 panel stats
(SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE) through the canonical
zenstats Rust panel binary via the vendored `zen_stats.py` shim.

**Source of truth:** `imazen/zenmetrics::zenstats`, called from
`zensim-validate::panel`, exposed as the `panel` CLI binary at
`~/work/zen/zensim/target/release/panel`. Python shim:
`~/work/zen/zensim/scripts/lib/zen_stats.py` (also vendored here).

## Decisions per file

Inventory found via:

```
grep -rln "^def srocc\|^def spearman\|^def pearson\|^def kendall\|^def pwrc\
\|^def outlier_ratio\|^def z_rmse\|^def plcc\|^def krocc\|np.corrcoef\
\|scipy.stats" --include="*.py" .
```

(scoped to in-tree files; `.claude/worktrees/*` agent worktrees are
not migrated — they're scratch.)

| File | Inline defs | Verdict |
|---|---|---|
| `zentrain/tools/zensim_metric_train.py` | `def srocc(a, b)` at L467, body `sstats.spearmanr` | **LEAVE** — scipy.stats wrapper, hot-loop per-epoch val callsite. scipy.stats IS the reference impl. |
| `zentrain/tools/correlation_cleanup.py` | `def spearman_corr_matrix(X)` at L158 | **LEAVE** — computes the p×p feature-correlation matrix used for column-cluster dedup. NOT a (pred, target) panel — `panel` has no matrix mode. Replacing with N² subprocess calls would be infeasible. |
| `benchmarks/feature_groups_2026-05-02.py` | `from scipy.stats import spearmanr` + `corr, _ = spearmanr(X, axis=0)` at L142 | **LEAVE** — matrix-form Spearman for hierarchical feature clustering. Different semantic (feature × feature corr matrix), not an IQA panel. scipy.stats is the reference impl. |
| `tools/zensim_per_class_consistency.py` | `from scipy.stats import spearmanr, pearsonr` + `rho, _ = spearmanr(z, sign*v)` at L137 | **LEAVE** — scipy.stats wrapper, called in a per-content-class metric-vs-metric comparison loop. Already at the reference impl. |
| `zentrain/tools/feature_transform_sweep.py` | `safe_pearson` L353, `safe_spearman` L391, `safe_z_rmse_score` L414 (mis-named — returns signed Pearson) | **LEAVE WITH CLARIFYING NOTE** — these compute |Pearson| / |Spearman| / signed-Pearson per (transformed_feature, bytes_log[:,cell]) for the picker transform screen. They are NOT IQA-panel reimpls; the screen iterates `n_features × n_transforms × n_param_grid × n_cells` and shelling to subprocess `panel` per inner call would be a ≥1000× slowdown. Add a comment pointing readers at `scripts/lib/zen_stats.py` for the IQA-panel case so nobody assumes these wear two hats. |

## Result

No inline reimpls of the Mohammadi-2025 panel stats survive in
zenanalyze in-tree code. The remaining usage of `np.corrcoef` /
`scipy.stats.{spearmanr,pearsonr}` is either:

1. A thin scipy.stats wrapper (already reference impl), or
2. Feature × feature correlation matrix work (different semantic
   from the (pred, target) panel zenstats covers), or
3. Inner-loop transform-screen helpers (different semantic — per-cell
   feature-vs-bytes_log, not pred-vs-MOS — performance-prohibitive to
   subprocess).

The vendored shim at `scripts/lib/zen_stats.py` is in place so any
**new** Python code that needs Mohammadi panel stats has a one-line
import.
