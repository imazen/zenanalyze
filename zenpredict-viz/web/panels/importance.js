// L0 importance panel — same 12-row heatmap + per-block stats + top/bottom lists.

import { featureLabel, blockStats } from '../feature_layout.js';

export function renderImportance(s, root) {
  root.innerHTML = '';
  const n = s.n_inputs;
  const vals = Array.from(s.l0_importance);

  // Distribution stats.
  const sorted = [...vals].sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))];
  root.insertAdjacentHTML('beforeend', `
    <div class="block-label">distribution · min ${sorted[0].toFixed(3)} · p25 ${q(0.25).toFixed(3)} · median ${q(0.5).toFixed(3)} · p75 ${q(0.75).toFixed(3)} · max ${sorted.at(-1).toFixed(3)}</div>
  `);

  // Heatmap, log-scaled so max doesn't drown out the rest.
  const logVals = vals.map(v => Math.log10(1 + v));
  const min = Math.min(...logVals);
  const max = Math.max(...logVals);
  const range = max - min || 1;
  const per_channel = n / 12;
  const grid = document.createElement('div');
  grid.className = 'heatmap';
  grid.style.gridTemplateColumns = `repeat(${per_channel}, var(--grid-cell))`;
  for (let col = 0; col < per_channel; col++) {
    for (let scale = 0; scale < 4; scale++) {
      for (let ch = 0; ch < 3; ch++) {
        const row = scale * 3 + ch;
        const featIdx = scale * 3 * per_channel + ch * per_channel + col;
        if (featIdx >= n) continue;
        const t = (logVals[featIdx] - min) / range;
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.style.background = importanceColor(t);
        cell.style.gridRow = `${row + 1}`;
        cell.style.gridColumn = `${col + 1}`;
        const info = featureLabel(featIdx, n);
        cell.title = `${info.label} (f${featIdx}) imp = ${vals[featIdx].toFixed(4)} · block ${info.block}`;
        grid.appendChild(cell);
      }
    }
  }
  root.appendChild(grid);

  // Per-block stats.
  const blocks = blockStats(vals, n);
  if (blocks.length > 0) {
    root.insertAdjacentHTML('beforeend', `
      <h3 style="margin: 20px 0 8px; font-size: 13px;">per-block L0 importance</h3>
      <table class="stats-table">
        <thead><tr><th>block</th><th>n</th><th>sum</th><th>mean</th><th>median</th><th>max</th><th>% total</th></tr></thead>
        <tbody>${blocks.map(b => `
          <tr><td class="label">${b.tag}</td><td>${b.n}</td><td>${b.sum.toFixed(2)}</td><td>${b.mean.toFixed(3)}</td><td>${b.median.toFixed(3)}</td><td>${b.max.toFixed(2)}</td><td>${b.pct.toFixed(2)}%</td></tr>
        `).join('')}</tbody>
      </table>
    `);
  }

  // Top + bottom 20.
  const indexed = vals.map((v, i) => ({ v, i, info: featureLabel(i, n) }));
  indexed.sort((a, b) => b.v - a.v);
  const top = indexed.slice(0, 20);
  const bottom = indexed.slice(-20).reverse();
  root.insertAdjacentHTML('beforeend', `
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 16px;">
      <div>
        <h3 style="margin: 0 0 8px; font-size: 13px;">top 20</h3>
        <table class="stats-table">
          <thead><tr><th>#</th><th>idx</th><th>label</th><th>imp</th></tr></thead>
          <tbody>${top.map((r, i) => `
            <tr><td class="label">${i + 1}</td><td>f${r.i}</td><td class="label">${r.info.label}</td><td>${r.v.toFixed(2)}</td></tr>
          `).join('')}</tbody>
        </table>
      </div>
      <div>
        <h3 style="margin: 0 0 8px; font-size: 13px;">bottom 20</h3>
        <table class="stats-table">
          <thead><tr><th>#</th><th>idx</th><th>label</th><th>imp</th></tr></thead>
          <tbody>${bottom.map((r, i) => `
            <tr><td class="label">${i + 1}</td><td>f${r.i}</td><td class="label">${r.info.label}</td><td>${r.v.toFixed(4)}</td></tr>
          `).join('')}</tbody>
        </table>
      </div>
    </div>
  `);
}

function importanceColor(t) {
  // black → orange → white for emphasis on hot features.
  const c = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, c * 2));
  const g = Math.round(255 * Math.max(0, c * 2 - 0.6));
  const b = Math.round(255 * Math.max(0, c * 2 - 1.2));
  return `rgb(${r},${g},${b})`;
}
