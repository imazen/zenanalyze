// Bake comparison panel (Track D MVP). Loads N bakes (≥2, ≤4),
// requires matching schemas (n_inputs / n_outputs / n_layers / per-layer
// dims), and renders:
//   - scaler shift heatmap (B - A) for mean and scale,
//   - L0 importance reshuffle (cell-by-cell delta),
//   - per-layer weight RMS comparison,
//   - calibration curve overlay when both bakes carry the same stages.

import { featureLabel, blockStats } from '../feature_layout.js';
import {
  decodeTanhOutputHead,
  applyTanhPin,
  decodeOutputCalibrationSpline,
  applySpline,
} from '../calibration_decoders.js';

const MAX_BAKES = 4;

export function renderCompare(parseBake, layerWeights, root) {
  root.innerHTML = '';
  const state = {
    bakes: [],   // [{ name, bytes, summary }]
  };

  const intro = document.createElement('div');
  intro.className = 'hint';
  intro.innerHTML = `
    Load 2&ndash;${MAX_BAKES} bakes with matching schemas (same n_inputs / n_outputs / layer
    dims). The first bake is the baseline; all others are compared against it.
    Per-codec & per-sample-α calibration overlays are best-effort — when the
    stage is present on both bakes with the same wire format, the panel draws
    both curves on one axis.
  `;
  root.appendChild(intro);

  const controls = document.createElement('div');
  controls.className = 'controls';
  controls.style.flexWrap = 'wrap';
  controls.innerHTML = `
    <input type="file" id="cmp-file" accept=".bin">
    <button class="action" id="cmp-clear">clear</button>
    <span class="block-label" id="cmp-status">load a bake to start.</span>
  `;
  root.appendChild(controls);

  const listEl = document.createElement('div');
  listEl.className = 'summary-list';
  listEl.style.cssText = 'font-family: var(--mono); font-size: 11px; margin-top: 8px; max-width: 540px;';
  root.appendChild(listEl);

  const body = document.createElement('div');
  body.style.marginTop = '16px';
  root.appendChild(body);

  controls.querySelector('#cmp-file').addEventListener('change', async (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    if (state.bakes.length >= MAX_BAKES) {
      alert(`maximum ${MAX_BAKES} bakes`);
      return;
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    let summary;
    try { summary = parseBake(bytes); }
    catch (err) { alert(`parse failed: ${err}`); return; }
    if (state.bakes.length > 0) {
      const baseline = state.bakes[0].summary;
      const mismatch = schemaMismatch(baseline, summary);
      if (mismatch) {
        alert(`schema mismatch vs baseline:\n${mismatch}`);
        return;
      }
    }
    state.bakes.push({ name: file.name, bytes, summary });
    renderList();
    if (state.bakes.length >= 2) renderDiff(body, state.bakes, layerWeights);
    ev.target.value = '';
  });
  controls.querySelector('#cmp-clear').addEventListener('click', () => {
    state.bakes = [];
    renderList();
    body.innerHTML = '';
  });

  function renderList() {
    listEl.innerHTML = '';
    if (state.bakes.length === 0) {
      listEl.innerHTML = '<div class="empty">no bakes loaded</div>';
      controls.querySelector('#cmp-status').textContent = 'load a bake to start.';
      return;
    }
    for (let i = 0; i < state.bakes.length; i++) {
      const b = state.bakes[i];
      const tag = i === 0 ? '<span style="color: var(--accent);">[baseline]</span>' : `<span style="color: var(--ok);">[Δ${i}]</span>`;
      listEl.insertAdjacentHTML('beforeend', `
        <div>${tag} ${b.name} · ${b.bytes.length} bytes · ${b.summary.n_inputs}→${b.summary.n_outputs} · ${b.summary.n_layers} layers</div>
      `);
    }
    controls.querySelector('#cmp-status').textContent = state.bakes.length < 2
      ? 'add at least one more bake to see the diff.'
      : `${state.bakes.length} bakes loaded.`;
  }
  renderList();
}

function schemaMismatch(a, b) {
  if (a.n_inputs !== b.n_inputs) return `n_inputs ${a.n_inputs} vs ${b.n_inputs}`;
  if (a.n_outputs !== b.n_outputs) return `n_outputs ${a.n_outputs} vs ${b.n_outputs}`;
  if (a.n_layers !== b.n_layers) return `n_layers ${a.n_layers} vs ${b.n_layers}`;
  for (let i = 0; i < a.n_layers; i++) {
    if (a.layers[i].in_dim !== b.layers[i].in_dim) return `L${i}.in_dim ${a.layers[i].in_dim} vs ${b.layers[i].in_dim}`;
    if (a.layers[i].out_dim !== b.layers[i].out_dim) return `L${i}.out_dim ${a.layers[i].out_dim} vs ${b.layers[i].out_dim}`;
  }
  return null;
}

function renderDiff(root, bakes, layerWeights) {
  root.innerHTML = '';
  const baseline = bakes[0];
  const n = baseline.summary.n_inputs;

  // ── Scaler shift heatmaps ───────────────────────────────────────
  root.insertAdjacentHTML('beforeend', '<h3 style="margin: 24px 0 8px; font-size: 13px;">scaler shift (Δ = bake[k] − baseline)</h3>');
  for (let k = 1; k < bakes.length; k++) {
    const dm = bakes[k].summary.scaler_mean.map((v, i) => v - baseline.summary.scaler_mean[i]);
    const ds = bakes[k].summary.scaler_scale.map((v, i) => v - baseline.summary.scaler_scale[i]);
    const wrap = document.createElement('div');
    wrap.innerHTML = `<div class="block-label">Δ${k}: ${bakes[k].name}</div>`;
    wrap.appendChild(makeShiftHeatmap('Δ mean', dm, n));
    wrap.appendChild(makeShiftHeatmap('Δ scale', ds, n));
    root.appendChild(wrap);
  }

  // ── L0 importance reshuffle ─────────────────────────────────────
  root.insertAdjacentHTML('beforeend', '<h3 style="margin: 24px 0 8px; font-size: 13px;">L0 importance reshuffle</h3>');
  for (let k = 1; k < bakes.length; k++) {
    const dimp = bakes[k].summary.l0_importance.map((v, i) => v - baseline.summary.l0_importance[i]);
    root.appendChild(makeShiftHeatmap(`Δ${k} L0 importance`, Array.from(dimp), n));
    // Top movers.
    const indexed = Array.from(dimp).map((v, i) => ({ v, i, info: featureLabel(i, n) }));
    indexed.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const movers = indexed.slice(0, 20);
    root.insertAdjacentHTML('beforeend', `
      <table class="stats-table">
        <thead><tr><th>#</th><th>idx</th><th>label</th><th>baseline</th><th>bake[${k}]</th><th>Δ</th></tr></thead>
        <tbody>${movers.map((m, j) => {
          const b = baseline.summary.l0_importance[m.i];
          const c = bakes[k].summary.l0_importance[m.i];
          const dir = m.v > 0 ? 'var(--ok)' : 'var(--bad)';
          return `<tr><td class="label">${j + 1}</td><td>f${m.i}</td><td class="label">${m.info.label}</td><td>${b.toFixed(3)}</td><td>${c.toFixed(3)}</td><td style="color: ${dir};">${m.v >= 0 ? '+' : ''}${m.v.toFixed(3)}</td></tr>`;
        }).join('')}</tbody>
      </table>
    `);
  }

  // ── Per-layer weight RMS ────────────────────────────────────────
  root.insertAdjacentHTML('beforeend', '<h3 style="margin: 24px 0 8px; font-size: 13px;">per-layer weight RMS</h3>');
  const rmsTable = document.createElement('table');
  rmsTable.className = 'stats-table';
  rmsTable.innerHTML = `
    <thead><tr><th>layer</th>${bakes.map((b, i) => `<th>${i === 0 ? 'baseline' : `bake[${i}]`}</th>`).join('')}<th>‖Δ‖</th></tr></thead>
    <tbody></tbody>
  `;
  const tbody = rmsTable.querySelector('tbody');
  const nLayers = baseline.summary.n_layers;
  for (let li = 0; li < nLayers; li++) {
    const cells = [];
    const allW = [];
    for (const bake of bakes) {
      try {
        const w = layerWeights(bake.bytes, li);
        const rms = Math.sqrt(Array.from(w).reduce((acc, v) => acc + v * v, 0) / w.length);
        cells.push(`<td>${rms.toFixed(4)}</td>`);
        allW.push(w);
      } catch (err) {
        cells.push(`<td style="color: var(--bad);">err</td>`);
        allW.push(null);
      }
    }
    // L2 difference vs baseline (taking the L2 norm of the weight diff
    // — same metric used for "did the layer change a lot").
    let normDelta = '—';
    if (bakes.length >= 2 && allW[0] && allW[1]) {
      const diff = allW[1].length === allW[0].length
        ? Math.sqrt(Array.from(allW[1]).reduce((acc, v, i) => acc + (v - allW[0][i]) ** 2, 0))
        : NaN;
      normDelta = Number.isFinite(diff) ? diff.toFixed(4) : '—';
    }
    tbody.insertAdjacentHTML('beforeend', `<tr><td class="label">L${li}</td>${cells.join('')}<td>${normDelta}</td></tr>`);
  }
  root.appendChild(rmsTable);

  // ── Calibration curve overlay ───────────────────────────────────
  const allHaveStage = (key) => bakes.every(b => b.summary.metadata_keys.some(m => m.key === key));
  if (allHaveStage('zentrain.tanh_output_head')) {
    root.appendChild(buildTanhOverlay(bakes));
  }
  if (allHaveStage('zentrain.output_calibration_spline')) {
    root.appendChild(buildSplineOverlay(bakes));
  }
}

function makeShiftHeatmap(title, vals, n_features) {
  const root = document.createElement('div');
  root.style.margin = '6px 0';
  const max = Math.max(...vals.map(v => Math.abs(v)), 1e-9);
  root.innerHTML = `<div class="block-label">${title} · max |Δ| = ${max.toFixed(4)}</div>`;
  const per_channel = n_features / 12;
  const grid = document.createElement('div');
  grid.className = 'heatmap';
  grid.style.gridTemplateColumns = `repeat(${per_channel}, var(--grid-cell))`;
  for (let col = 0; col < per_channel; col++) {
    for (let scale = 0; scale < 4; scale++) {
      for (let ch = 0; ch < 3; ch++) {
        const row = scale * 3 + ch;
        const featIdx = scale * 3 * per_channel + ch * per_channel + col;
        if (featIdx >= n_features) continue;
        const v = vals[featIdx];
        const t = (v / max + 1) / 2;
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.style.background = divergingColor(t);
        cell.style.gridRow = `${row + 1}`;
        cell.style.gridColumn = `${col + 1}`;
        const info = featureLabel(featIdx, n_features);
        cell.title = `${info.label} (f${featIdx})\nΔ = ${v.toFixed(6)}`;
        grid.appendChild(cell);
      }
    }
  }
  root.appendChild(grid);
  return root;
}

function divergingColor(t) {
  const c = Math.max(0, Math.min(1, t));
  if (c < 0.5) {
    const k = c * 2;
    const r = Math.round(100 + 155 * k);
    const g = Math.round(140 + 115 * k);
    const b = Math.round(220 + 35 * k);
    return `rgb(${r},${g},${b})`;
  }
  const k = (c - 0.5) * 2;
  const r = Math.round(255);
  const g = Math.round(255 - 175 * k);
  const b = Math.round(255 - 175 * k);
  return `rgb(${r},${g},${b})`;
}

function buildTanhOverlay(bakes) {
  const wrap = document.createElement('div');
  wrap.innerHTML = '<h3 style="margin: 24px 0 8px; font-size: 13px;">tanh_output_head overlay</h3>';
  const scales = bakes.map(b => {
    const e = b.summary.metadata_keys.find(m => m.key === 'zentrain.tanh_output_head');
    return decodeTanhOutputHead(e.value_hex).scale;
  });
  const sMax = Math.max(...scales.filter(Number.isFinite));
  const xMin = -3 * sMax;
  const xMax = 3 * sMax;
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 220;
  canvas.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line);';
  wrap.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  drawAxes(ctx, canvas, xMin, xMax, 0, 100, 'y_pre', 'score');
  const palette = ['#6cb6ff', '#f6a96a', '#7ee787', '#ff7b72'];
  let labelY = 14;
  for (let k = 0; k < bakes.length; k++) {
    ctx.strokeStyle = palette[k % palette.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let px = 0; px < canvas.width; px++) {
      const x = xMin + (xMax - xMin) * (px / canvas.width);
      const y = applyTanhPin(x, scales[k]);
      const py = canvas.height - (y / 100) * canvas.height;
      if (px === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.fillStyle = palette[k % palette.length];
    ctx.font = '11px monospace';
    ctx.fillText(`${k === 0 ? 'baseline' : `bake[${k}]`} scale=${scales[k].toFixed(4)}`, 8, labelY);
    labelY += 14;
  }
  return wrap;
}

function buildSplineOverlay(bakes) {
  const wrap = document.createElement('div');
  wrap.innerHTML = '<h3 style="margin: 24px 0 8px; font-size: 13px;">output_calibration_spline overlay</h3>';
  const splines = bakes.map(b => {
    const e = b.summary.metadata_keys.find(m => m.key === 'zentrain.output_calibration_spline');
    return decodeOutputCalibrationSpline(e.value_hex);
  });
  if (splines.some(s => s.error)) {
    wrap.insertAdjacentHTML('beforeend', `<div class="hint" style="color: var(--bad);">decode error in one or more bakes — skipping overlay</div>`);
    return wrap;
  }
  const xMin = Math.min(...splines.map(s => s.xs[0]));
  const xMax = Math.max(...splines.map(s => s.xs[s.xs.length - 1]));
  const yLo = Math.min(...splines.map(s => Math.min(...s.ys)));
  const yHi = Math.max(...splines.map(s => Math.max(...s.ys)));
  const canvas = document.createElement('canvas');
  canvas.width = 720;
  canvas.height = 260;
  canvas.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line);';
  wrap.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  drawAxes(ctx, canvas, xMin, xMax, yLo, yHi, 'pinned score', 'calibrated score');
  const palette = ['#6cb6ff', '#f6a96a', '#7ee787', '#ff7b72'];
  let labelY = 14;
  for (let k = 0; k < bakes.length; k++) {
    const s = splines[k];
    ctx.strokeStyle = palette[k % palette.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let px = 0; px < canvas.width; px++) {
      const x = xMin + (xMax - xMin) * (px / canvas.width);
      const y = applySpline(x, s);
      const py = canvas.height - ((y - yLo) / (yHi - yLo)) * canvas.height;
      if (px === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.fillStyle = palette[k % palette.length];
    ctx.font = '11px monospace';
    ctx.fillText(`${k === 0 ? 'baseline' : `bake[${k}]`} ${s.xs.length} knots`, 8, labelY);
    labelY += 14;
  }
  return wrap;
}

function drawAxes(ctx, canvas, xMin, xMax, yMin, yMax, xLabel, yLabel) {
  ctx.fillStyle = '#181b22';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#2a2f3a';
  for (let i = 0; i <= 10; i++) {
    const x = (i / 10) * canvas.width;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let i = 0; i <= 5; i++) {
    const y = (i / 5) * canvas.height;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }
  ctx.fillStyle = '#9aa3b2';
  ctx.font = '11px monospace';
  ctx.fillText(`${xLabel} ∈ [${xMin.toFixed(2)}, ${xMax.toFixed(2)}]`, 8, canvas.height - 6);
  ctx.fillText(`${yLabel} ∈ [${yMin.toFixed(2)}, ${yMax.toFixed(2)}]`, 8, 14);
}
