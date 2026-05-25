// Scaler panel — per-feature (mean, scale) heatmap, 12 rows × N cols.

import { featureLabel, cellLabels } from '../feature_layout.js';

export function renderScaler(s, root) {
  root.innerHTML = '';
  const n = s.n_inputs;
  // Layout: 12 rows (4 scales × 3 channels), n/12 cols.
  const rows = 12;
  if (n % rows !== 0) {
    root.innerHTML = `<div class="hint">feature count ${n} is not a multiple of 12; raw layout view only.</div>`;
  }
  const cols = Math.ceil(n / rows);

  for (const [title, vals] of [['scaler_mean', s.scaler_mean], ['scaler_scale', s.scaler_scale]]) {
    root.insertAdjacentHTML('beforeend', `<div class="block-label">${title} · range [${Math.min(...vals).toFixed(3)}, ${Math.max(...vals).toFixed(3)}]</div>`);
    const grid = document.createElement('div');
    grid.className = 'heatmap';
    grid.style.gridTemplateColumns = `repeat(${cols}, var(--grid-cell))`;
    // values come in feature-major order (f0..fN). Build cells in
    // column-major (col = feature_block_idx, row = scale*3 + channel).
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min || 1;
    // We treat features as: idx = scale * (3 * per_channel) + channel * per_channel + feature_in_channel.
    // The 12 (scale, channel) cells are rows. Per-channel features are columns.
    const per_channel = n / 12;
    for (let col = 0; col < per_channel; col++) {
      for (let scale = 0; scale < 4; scale++) {
        for (let ch = 0; ch < 3; ch++) {
          const row = scale * 3 + ch;
          const featIdx = scale * 3 * per_channel + ch * per_channel + col;
          if (featIdx >= n) continue;
          const v = vals[featIdx];
          const t = (v - min) / range;
          const cell = document.createElement('div');
          cell.className = 'cell';
          cell.style.background = colorMap(t);
          cell.style.gridRow = `${row + 1}`;
          cell.style.gridColumn = `${col + 1}`;
          cell.dataset.idx = featIdx;
          cell.dataset.val = v;
          const info = featureLabel(featIdx, n);
          const tooltipParts = [
            `${info.label} (f${featIdx})`,
            `${title}: ${v.toFixed(4)}`,
          ];
          if (info.math_summary) tooltipParts.push(`math: ${info.math_summary}`);
          if (info.source_ref) tooltipParts.push(`src: ${info.source_ref}`);
          cell.title = tooltipParts.join('\n');
          grid.appendChild(cell);
        }
      }
    }
    root.appendChild(grid);
  }
}

function colorMap(t) {
  // viridis-ish: dark purple → teal → yellow.
  const c = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.pow(c, 1.5));
  const g = Math.round(255 * c);
  const b = Math.round(255 * Math.pow(1 - c, 1.5) * 0.7);
  return `rgb(${r},${g},${b})`;
}
