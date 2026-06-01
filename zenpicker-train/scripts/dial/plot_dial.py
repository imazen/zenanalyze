import pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows = pickle.load(open('/mnt/v/zen/dial-quality-2026-06-01/oracle_frontier.pkl','rb'))
T = [r[0] for r in rows]
ob = [r[2] for r in rows]   # oracle byte pctl dicts
qe = [r[4] for r in rows]   # qhit-err pctl dicts
zr = [r[5] for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(13,5))
# Plot A: byte cost to hit target — percentile bands (heavy-tailed → log y)
for key,lab,c in [('p50','median','C0'),('p75','p75','C1'),('p90','p90','C2'),('p95','p95','C3')]:
    ax[0].plot(T, [d[key] for d in ob], marker='o', label=lab, color=c)
ax[0].fill_between(T, [d['p50'] for d in ob], [d['p95'] for d in ob], alpha=0.08, color='C0')
ax[0].set_yscale('log'); ax[0].set_xlabel('Profile-A target (zensim:a)'); ax[0].set_ylabel('encoded bytes (least-waste optimum)')
ax[0].set_title('Byte cost to hit a quality target\n(cross-image spread, 320 imgs × 36 cells × 29 q)')
ax[0].grid(True, which='both', alpha=0.3); ax[0].legend(title='percentile')
# Plot B: dial precision — quality-hit error mean±std + RMSE
m=[d['mean'] for d in qe]; s=[d['std'] for d in qe]
ax[1].errorbar(T, m, yerr=s, marker='o', capsize=3, color='C0', label='achieved − target (mean ± std)')
ax[1].plot(T, zr, marker='s', color='C3', label='RMSE')
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('Profile-A target (zensim:a)'); ax[1].set_ylabel('quality-hit error (zensim:a points)')
ax[1].set_title('Dial precision vs target\n(q-granularity: tight at high-q, coarse at low-q)')
ax[1].grid(True, alpha=0.3); ax[1].legend()
plt.tight_layout(); plt.savefig('/mnt/v/zen/dial-quality-2026-06-01/dial_frontier.png', dpi=110)
print("wrote dial_frontier.png")
