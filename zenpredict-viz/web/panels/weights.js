// Per-layer weights heatmap panel (Track C P1).
// Renders W [in_dim, out_dim] as a Canvas-based heatmap (much faster
// than D3 SVG for the dense 372×128 matrices that v_tuner_v9 has).
// A bias strip + i8-scales strip sit alongside.

import { featureLabel } from '../feature_layout.js';

const CANVAS_MAX_WIDTH = 1200;
const CANVAS_MAX_HEIGHT = 800;
const MAX_CELLS_PER_DIM = 1024;

export function renderWeights(summary, getLayerWeights, root) {
  root.innerHTML = '';
  const layers = summary.layers;
  if (!layers.length) {
    root.innerHTML = '<div class="empty">no layers found in this bake.</div>';
    return;
  }

  // Layer-selector + display options.
  const controls = document.createElement('div');
  controls.className = 'controls';
  controls.innerHTML = `
    <label style="font-size: 11px; color: var(--fg-dim);">layer:</label>
    <select id="wt-layer">
      ${layers.map(l => `<option value="${l.idx}">L${l.idx} · ${l.activation} · ${l.in_dim}×${l.out_dim} · ${l.dtype}</option>`).join('')}
    </select>
    <label style="font-size: 11px; color: var(--fg-dim); margin-left: 12px;">symmetric scale:</label>
    <input type="checkbox" id="wt-symm" checked>
  `;
  root.appendChild(controls);

  const statusLine = document.createElement('div');
  statusLine.className = 'block-label';
  statusLine.id = 'wt-status';
  statusLine.textContent = 'rendering...';
  root.appendChild(statusLine);

  const grid = document.createElement('div');
  grid.style.cssText = 'display: grid; grid-template-columns: 1fr 80px 80px; gap: 8px; margin-top: 8px;';
  root.appendChild(grid);

  const heatmapBox = document.createElement('div');
  heatmapBox.innerHTML = '<div class="block-label">W · weight matrix (rows = inputs, cols = hidden units)</div>';
  const heatmapCanvas = document.createElement('canvas');
  heatmapCanvas.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line); cursor: crosshair; image-rendering: pixelated;';
  heatmapBox.appendChild(heatmapCanvas);
  const cellTooltip = document.createElement('div');
  cellTooltip.style.cssText = 'font-family: var(--mono); font-size: 11px; color: var(--fg-dim); min-height: 1.4em; margin-top: 4px;';
  cellTooltip.textContent = 'hover the heatmap to read a cell value';
  heatmapBox.appendChild(cellTooltip);
  grid.appendChild(heatmapBox);

  const biasBox = document.createElement('div');
  biasBox.innerHTML = '<div class="block-label">B · biases</div>';
  const biasCanvas = document.createElement('canvas');
  biasCanvas.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line); width: 100%; height: auto;';
  biasBox.appendChild(biasCanvas);
  grid.appendChild(biasBox);

  const scaleBox = document.createElement('div');
  scaleBox.innerHTML = '<div class="block-label">i8 scales</div>';
  const scaleCanvas = document.createElement('canvas');
  scaleCanvas.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line); width: 100%; height: auto;';
  scaleBox.appendChild(scaleCanvas);
  grid.appendChild(scaleBox);

  const histogramBox = document.createElement('div');
  histogramBox.style.cssText = 'margin-top: 16px;';
  root.appendChild(histogramBox);

  const select = controls.querySelector('#wt-layer');
  const symmCheck = controls.querySelector('#wt-symm');
  const renderForSelectedLayer = () => {
    const idx = parseInt(select.value, 10);
    renderLayer(idx);
  };
  select.addEventListener('change', renderForSelectedLayer);
  symmCheck.addEventListener('change', renderForSelectedLayer);

  function renderLayer(idx) {
    const layer = layers[idx];
    statusLine.textContent = `L${idx} · ${layer.in_dim}×${layer.out_dim} cells = ${(layer.in_dim * layer.out_dim).toLocaleString()} · weight range [${layer.weight_min.toFixed(4)}, ${layer.weight_max.toFixed(4)}] · bias range [${layer.bias_min.toFixed(4)}, ${layer.bias_max.toFixed(4)}]`;
    let w;
    try {
      w = getLayerWeights(idx);
    } catch (err) {
      statusLine.textContent = `error fetching weights: ${err}`;
      return;
    }
    const arr = Array.from(w);
    renderHeatmap(heatmapCanvas, cellTooltip, arr, layer, symmCheck.checked, summary.n_inputs);
    renderStrip(biasCanvas, layer.biases ?? [], layer.bias_min, layer.bias_max, symmCheck.checked);
    if (layer.i8_scales) {
      renderStrip(scaleCanvas, layer.i8_scales, Math.min(...layer.i8_scales), Math.max(...layer.i8_scales), false);
      scaleBox.style.display = '';
    } else {
      scaleBox.style.display = 'none';
    }
    renderHistogram(histogramBox, arr, layer);
  }

  // Note: we use summary's per-layer bias_min/max/etc; the biases vector
  // itself isn't included in the summary (it's a stat-only view), so we
  // synthesise a strip from the bias-stats. We could pull them via a new
  // WASM export, but for an MVP "what does the bias look like" the
  // min/mean/max strip is enough — the bias panel is just a sanity
  // visualisation.
  for (const l of layers) l.biases = null;  // signal "use stats only"

  renderForSelectedLayer();
}

function renderHeatmap(canvas, tooltip, w, layer, symm, n_inputs) {
  const { in_dim, out_dim } = layer;
  const widthCells = Math.min(out_dim, MAX_CELLS_PER_DIM);
  const heightCells = Math.min(in_dim, MAX_CELLS_PER_DIM);
  const cellW = Math.max(1, Math.floor(CANVAS_MAX_WIDTH / widthCells));
  const cellH = Math.max(1, Math.floor(CANVAS_MAX_HEIGHT / heightCells));
  canvas.width = widthCells * cellW;
  canvas.height = heightCells * cellH;
  const ctx = canvas.getContext('2d');
  let mn, mx;
  if (symm) {
    const m = Math.max(Math.abs(layer.weight_min), Math.abs(layer.weight_max), 1e-12);
    mn = -m;
    mx = m;
  } else {
    mn = layer.weight_min;
    mx = layer.weight_max;
  }
  const range = mx - mn || 1;
  const img = ctx.createImageData(canvas.width, canvas.height);
  // Stride-and-downsample if in_dim > MAX_CELLS_PER_DIM.
  for (let y = 0; y < heightCells; y++) {
    const iSrc = Math.floor(y * (in_dim / heightCells));
    for (let x = 0; x < widthCells; x++) {
      const oSrc = Math.floor(x * (out_dim / widthCells));
      const v = w[iSrc * out_dim + oSrc];
      const t = (v - mn) / range;
      const [r, g, b] = symm ? divergingColor(t) : sequentialColor(t);
      for (let dy = 0; dy < cellH; dy++) {
        for (let dx = 0; dx < cellW; dx++) {
          const px = ((y * cellH + dy) * canvas.width + x * cellW + dx) * 4;
          img.data[px] = r;
          img.data[px + 1] = g;
          img.data[px + 2] = b;
          img.data[px + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const cx = Math.floor((ev.clientX - rect.left) * canvas.width / rect.width);
    const cy = Math.floor((ev.clientY - rect.top) * canvas.height / rect.height);
    const x = Math.floor(cx / cellW);
    const y = Math.floor(cy / cellH);
    if (x < 0 || x >= widthCells || y < 0 || y >= heightCells) return;
    const iSrc = Math.floor(y * (in_dim / heightCells));
    const oSrc = Math.floor(x * (out_dim / widthCells));
    const v = w[iSrc * out_dim + oSrc];
    let inputLabel = `i${iSrc}`;
    if (layer.idx === 0) {
      const info = featureLabel(iSrc, n_inputs);
      inputLabel = `${info.label} (f${iSrc})`;
    }
    tooltip.textContent = `W[${inputLabel}, h${oSrc}] = ${v.toFixed(6)}`;
  };
  canvas.onmouseleave = () => { tooltip.textContent = 'hover the heatmap to read a cell value'; };
}

function renderStrip(canvas, vals, mn, mx, symm) {
  const n = vals.length;
  if (n === 0) {
    canvas.width = 1;
    canvas.height = 1;
    return;
  }
  const cellH = Math.max(1, Math.floor(800 / n));
  canvas.width = 60;
  canvas.height = n * cellH;
  const ctx = canvas.getContext('2d');
  let lo, hi;
  if (symm) {
    const m = Math.max(Math.abs(mn), Math.abs(mx), 1e-12);
    lo = -m;
    hi = m;
  } else {
    lo = mn;
    hi = mx;
  }
  const range = hi - lo || 1;
  for (let i = 0; i < n; i++) {
    const t = (vals[i] - lo) / range;
    const [r, g, b] = symm ? divergingColor(t) : sequentialColor(t);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, i * cellH, canvas.width, cellH);
  }
}

function renderHistogram(root, vals, layer) {
  // Simple 80-bin histogram of weight values.
  root.innerHTML = '<div class="block-label">weight value histogram (80 bins)</div>';
  const bins = 80;
  const counts = new Array(bins).fill(0);
  const mn = layer.weight_min, mx = layer.weight_max;
  const range = mx - mn || 1;
  for (const v of vals) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor((v - mn) / range * bins)));
    counts[b]++;
  }
  const maxC = Math.max(...counts);
  const canvas = document.createElement('canvas');
  canvas.width = bins * 6;
  canvas.height = 100;
  canvas.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line);';
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#6cb6ff';
  for (let i = 0; i < bins; i++) {
    const h = Math.round((counts[i] / maxC) * canvas.height);
    ctx.fillRect(i * 6, canvas.height - h, 5, h);
  }
  root.appendChild(canvas);
  root.insertAdjacentHTML('beforeend', `
    <div class="block-label" style="margin-top: 4px;">[${mn.toFixed(4)} ... ${mx.toFixed(4)}], n = ${vals.length.toLocaleString()}, mean = ${layer.weight_mean.toFixed(4)}</div>
  `);
}

function divergingColor(t) {
  const c = Math.max(0, Math.min(1, t));
  // Red ↔ white ↔ blue (diverging at 0.5).
  if (c < 0.5) {
    const k = c * 2;
    const r = Math.round(80 + (255 - 80) * k);
    const g = Math.round(80 + (255 - 80) * k);
    const b = Math.round(180 + (255 - 180) * k);
    return [r, g, b];
  }
  const k = (c - 0.5) * 2;
  const r = Math.round(255 - (255 - 200) * k);
  const g = Math.round(255 - (255 - 80) * k);
  const b = Math.round(255 - (255 - 80) * k);
  return [r, g, b];
}

function sequentialColor(t) {
  const c = Math.max(0, Math.min(1, t));
  return [
    Math.round(255 * c),
    Math.round(255 * c * 0.6),
    Math.round(255 * (1 - c) * 0.4),
  ];
}
