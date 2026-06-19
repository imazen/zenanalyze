import pyarrow.parquet as pq, numpy as np
from scipy.stats import rankdata

PARQ="/mnt/v/output/imazen-26-features/imazen26_features_2026-06-13.parquet"
t = pq.read_table(PARQ)
cols = [c for c in t.column_names if c.startswith("feat_")]

# group percentile columns by base distribution
import re
def base(c):
    m = re.match(r"(feat_(?:laplacian_variance|aq_map|noise_floor_y|noise_floor_uv|quant_survival_y|quant_survival_uv))_?(p\d+|peak)?$", c)
    return m.group(1) if (m and m.group(2)) else None
groups = {}
for c in cols:
    b = base(c)
    if b: groups.setdefault(b, []).append(c)

# Spearman = Pearson on ranks. Sample 60k rows for speed.
n = t.num_rows
idx = np.linspace(0, n-1, min(60000, n)).astype(int)
def col(c):
    a = np.asarray(t.column(c))[idx].astype(np.float64)
    a = np.nan_to_num(a, nan=np.nanmedian(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.0)
    return rankdata(a)

print("=== per-base-distribution percentile redundancy (Spearman) ===")
print(f"{'base distribution':<28} {'#perc':>5} {'min|ρ|adj':>9} {'mean|ρ|':>8} {'eff-rank@95%':>12} {'eff-rank@99%':>12}")
for b, cs in sorted(groups.items()):
    cs = sorted(cs, key=lambda c:(len(c), c))
    R = np.vstack([col(c) for c in cs])            # (k, N) ranks
    C = np.corrcoef(R)                             # Spearman matrix
    k = len(cs)
    # adjacent-percentile correlations (off-diagonal neighbors)
    offdiag = C[np.triu_indices(k,1)]
    # effective rank via eigenvalues of the correlation matrix
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    cumvar = np.cumsum(ev)/k
    er95 = int(np.searchsorted(cumvar, 0.95)+1)
    er99 = int(np.searchsorted(cumvar, 0.99)+1)
    print(f"{b.replace('feat_',''):<28} {k:>5} {offdiag.min():>9.3f} {np.abs(offdiag).mean():>8.3f} {er95:>12} {er99:>12}")

print("\n=== cross-distribution SAME-percentile correlation (the perfect-Jaccard claim) ===")
for p in ["p50","p75","p90"]:
    members = [f"feat_{d}_{p}" for d in ["aq_map","noise_floor_y","quant_survival_y"] if f"feat_{d}_{p}" in cols]
    if len(members)>=2:
        R = np.vstack([col(c) for c in members]); C=np.corrcoef(R)
        off = C[np.triu_indices(len(members),1)]
        print(f"  {p}: {[m.replace('feat_','') for m in members]}  ρ range [{off.min():.3f}, {off.max():.3f}]")

print("\n=== GLOBAL dimensionality of the whole percentile space ===")
allperc = sorted([c for c in cols if base(c)])
R = np.vstack([col(c) for c in allperc]); C=np.corrcoef(R)
k=len(allperc); ev=np.sort(np.linalg.eigvalsh(C))[::-1]; cum=np.cumsum(ev)/k
for thr in (0.90,0.95,0.99):
    print(f"  {len(allperc)} percentile features -> eff-rank @{int(thr*100)}% = {int(np.searchsorted(cum,thr)+1)}")
print("  => that many independent components span ALL existing percentile sweeps.")
