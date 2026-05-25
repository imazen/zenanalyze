// L0 importance panel — 12-row heatmap + per-block stats + top/bottom
// lists + (Track B) feature search box.

import { featureLabel, blockStats, searchFeatures } from '../feature_layout.js';

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

  // Search box — filters the top/bottom lists below.
  const searchRow = document.createElement('div');
  searchRow.className = 'controls';
  searchRow.innerHTML = `
    <label style="font-size: 11px; color: var(--fg-dim);">filter:</label>
    <input type="text" id="imp-search" placeholder="e.g. iw_ssim, s2.Cb, ringing, hf_energy_loss" style="max-width: 360px;">
    <span class="block-label" id="imp-search-count" style="margin: 0 0 0 8px;"></span>
  `;
  root.appendChild(searchRow);

  // Heatmap, log-scaled so max doesn't drown out the rest.
  const logVals = vals.map(v => Math.log10(1 + v));
  const min = Math.min(...logVals);
  const max = Math.max(...logVals);
  const range = max - min || 1;
  const per_channel = n / 12;
  const grid = document.createElement('div');
  grid.className = 'heatmap';
  grid.style.gridTemplateColumns = `repeat(${per_channel}, var(--grid-cell))`;
  const cellByIdx = new Map();
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
        const tooltipParts = [
          `${info.label} (f${featIdx})`,
          `block: ${info.block}`,
          `imp: ${vals[featIdx].toFixed(4)}`,
        ];
        if (info.math_summary) tooltipParts.push(`math: ${info.math_summary}`);
        if (info.source_ref) tooltipParts.push(`src: ${info.source_ref}`);
        cell.title = tooltipParts.join('\n');
        grid.appendChild(cell);
        cellByIdx.set(featIdx, cell);
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

  // Top + bottom 20 (filterable by the search box).
  const indexed = vals.map((v, i) => ({ v, i, info: featureLabel(i, n) }));
  const listsContainer = document.createElement('div');
  listsContainer.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 16px;';
  root.appendChild(listsContainer);
  renderLists(listsContainer, indexed, null);

  // Wire search.
  const searchInput = root.querySelector('#imp-search');
  const searchCount = root.querySelector('#imp-search-count');
  searchInput.addEventListener('input', () => {
    const matches = searchFeatures(searchInput.value, n);
    renderLists(listsContainer, indexed, matches);
    // Dim non-matching cells in the heatmap.
    if (matches === null) {
      cellByIdx.forEach(c => c.style.opacity = '1');
      searchCount.textContent = '';
    } else {
      const matchSet = new Set(matches);
      cellByIdx.forEach((c, i) => c.style.opacity = matchSet.has(i) ? '1' : '0.15');
      searchCount.textContent = `${matches.length} match${matches.length === 1 ? '' : 'es'}`;
    }
  });
}

function renderLists(container, indexed, matches) {
  const filtered = matches === null ? indexed : indexed.filter(e => matches.includes(e.i));
  const sorted = [...filtered].sort((a, b) => b.v - a.v);
  const top = sorted.slice(0, 20);
  const bottom = sorted.slice(-20).reverse();
  container.innerHTML = `
    <div>
      <h3 style="margin: 0 0 8px; font-size: 13px;">top 20${matches !== null ? ' (filtered)' : ''}</h3>
      <table class="stats-table">
        <thead><tr><th>#</th><th>idx</th><th>label</th><th>imp</th></tr></thead>
        <tbody>${top.map((r, i) => `
          <tr title="${(r.info.math_summary || '').replace(/"/g, '&quot;')}"><td class="label">${i + 1}</td><td>f${r.i}</td><td class="label">${r.info.label}</td><td>${r.v.toFixed(2)}</td></tr>
        `).join('')}</tbody>
      </table>
    </div>
    <div>
      <h3 style="margin: 0 0 8px; font-size: 13px;">bottom 20${matches !== null ? ' (filtered)' : ''}</h3>
      <table class="stats-table">
        <thead><tr><th>#</th><th>idx</th><th>label</th><th>imp</th></tr></thead>
        <tbody>${bottom.map((r, i) => `
          <tr title="${(r.info.math_summary || '').replace(/"/g, '&quot;')}"><td class="label">${i + 1}</td><td>f${r.i}</td><td class="label">${r.info.label}</td><td>${r.v.toFixed(4)}</td></tr>
        `).join('')}</tbody>
      </table>
    </div>
  `;
}

function importanceColor(t) {
  // black → orange → white for emphasis on hot features.
  const c = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, c * 2));
  const g = Math.round(255 * Math.max(0, c * 2 - 0.6));
  const b = Math.round(255 * Math.max(0, c * 2 - 1.2));
  return `rgb(${r},${g},${b})`;
}
