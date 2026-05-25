// Calibration panel (Track C P1) — renders the post-MLP zensim
// calibration stages whose metadata is present on the loaded bake:
//
//   - zentrain.tanh_output_head         → 100·σ(x/scale) curve
//   - zentrain.output_calibration_spline → PCHIP spline with knot dots
//   - zentrain.per_codec_calibration    → one affine line per codec
//   - zentrain.per_sample_alpha_head    → architecture diagram (no curve)
//
// Stages absent from the bake render a "(not present)" placeholder so
// users see the full ordered list and know what's missing.

import {
  decodeTanhOutputHead,
  applyTanhPin,
  decodeOutputCalibrationSpline,
  applySpline,
  decodePerCodecCalibration,
  decodePerSampleAlphaHead,
} from '../calibration_decoders.js';

export function renderCalibration(summary, root) {
  root.innerHTML = '';
  const md = new Map(summary.metadata_keys.map(m => [m.key, m]));

  // Order matters — match the runtime pipeline order in
  // zensim::metric::apply_mlp_scoring_with_codec.
  const stages = [
    {
      key: 'zentrain.tanh_output_head',
      label: 'tanh output head (sigmoid pin)',
      summary: '100·σ(y_pre / scale) — pins MLP output to [0, 100] without a post-hoc affine.',
    },
    {
      key: 'zentrain.output_calibration_spline',
      label: 'output calibration spline (PCHIP)',
      summary: 'Monotone cubic Hermite spline on the tanh-pinned score. Calibrates cross-distortion bias toward JND landmarks.',
    },
    {
      key: 'zentrain.per_codec_calibration',
      label: 'per-codec calibration',
      summary: 'After-spline affine (alpha + beta·s) per codec — within-codec rank-preserving (beta > 0); aligns cross-codec means.',
    },
    {
      key: 'zentrain.per_sample_alpha_head',
      label: 'per-sample-α head',
      summary: 'Mixes a rank head (dot(rank_w, h)) with a pool head (p-norm reducer over hidden-vector stats) via a per-sample sigmoid gate.',
    },
  ];

  for (const stage of stages) {
    const entry = md.get(stage.key);
    const section = document.createElement('div');
    section.style.cssText = 'margin-bottom: 24px; border-left: 3px solid var(--line); padding-left: 12px;';
    if (!entry) {
      section.innerHTML = `
        <div class="block-label" style="color: var(--fg-dim);">${stage.label} <span style="color: var(--bad);">(not present)</span></div>
        <div class="hint">${stage.summary}</div>
      `;
      root.appendChild(section);
      continue;
    }
    section.style.borderLeftColor = 'var(--ok)';
    section.innerHTML = `
      <div class="block-label" style="color: var(--ok);">${stage.label} <span style="color: var(--fg-dim);">· ${entry.value_len} bytes</span></div>
      <div class="hint">${stage.summary}</div>
    `;
    root.appendChild(section);
    const body = document.createElement('div');
    body.style.marginTop = '8px';
    section.appendChild(body);

    switch (stage.key) {
      case 'zentrain.tanh_output_head':
        renderTanh(entry, body);
        break;
      case 'zentrain.output_calibration_spline':
        renderSpline(entry, body);
        break;
      case 'zentrain.per_codec_calibration':
        renderPerCodec(entry, body);
        break;
      case 'zentrain.per_sample_alpha_head': {
        const n_hidden = summary.layers.length > 1
          ? summary.layers[summary.layers.length - 1].in_dim
          : summary.layers[0].out_dim;
        renderAlphaHead(entry, body, n_hidden);
        break;
      }
    }
  }
}

function renderTanh(entry, root) {
  const r = decodeTanhOutputHead(entry.value_hex);
  if (r.error) {
    root.innerHTML = `<div class="hint" style="color: var(--bad);">decode error: ${r.error}</div>`;
    return;
  }
  root.innerHTML = `<div class="block-label">scale = ${r.scale.toFixed(6)}</div>`;
  // Plot 100·σ(x/scale) for x ∈ [-3*scale, 3*scale].
  const canvas = document.createElement('canvas');
  canvas.width = 600;
  canvas.height = 220;
  canvas.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line); margin-top: 8px;';
  root.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const xMin = -3 * r.scale;
  const xMax = 3 * r.scale;
  drawAxes(ctx, canvas, xMin, xMax, 0, 100, 'y_pre', 'score');
  ctx.strokeStyle = '#6cb6ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let px = 0; px < canvas.width; px++) {
    const x = xMin + (xMax - xMin) * (px / canvas.width);
    const y = applyTanhPin(x, r.scale);
    const py = canvas.height - (y / 100) * canvas.height;
    if (px === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function renderSpline(entry, root) {
  const r = decodeOutputCalibrationSpline(entry.value_hex);
  if (r.error) {
    root.innerHTML = `<div class="hint" style="color: var(--bad);">decode error: ${r.error}</div>`;
    return;
  }
  const { xs, ys, derivs } = r;
  const xMin = xs[0];
  const xMax = xs[xs.length - 1];
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const yRange = (yMax - yMin) || 1;
  const yLo = yMin - 0.05 * yRange;
  const yHi = yMax + 0.05 * yRange;
  root.innerHTML = `<div class="block-label">${xs.length} knots · x ∈ [${xMin.toFixed(2)}, ${xMax.toFixed(2)}], y ∈ [${yMin.toFixed(2)}, ${yMax.toFixed(2)}]</div>`;
  const canvas = document.createElement('canvas');
  canvas.width = 720;
  canvas.height = 260;
  canvas.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line); margin-top: 8px;';
  root.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  drawAxes(ctx, canvas, xMin, xMax, yLo, yHi, 'pinned score', 'calibrated score');
  // Curve.
  ctx.strokeStyle = '#7ee787';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let px = 0; px < canvas.width; px++) {
    const x = xMin + (xMax - xMin) * (px / canvas.width);
    const y = applySpline(x, r);
    const py = canvas.height - ((y - yLo) / (yHi - yLo)) * canvas.height;
    if (px === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  // Knot dots.
  ctx.fillStyle = '#f6a96a';
  for (let i = 0; i < xs.length; i++) {
    const px = ((xs[i] - xMin) / (xMax - xMin)) * canvas.width;
    const py = canvas.height - ((ys[i] - yLo) / (yHi - yLo)) * canvas.height;
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
  // Knot table.
  const tab = document.createElement('table');
  tab.className = 'stats-table';
  tab.style.marginTop = '12px';
  tab.innerHTML = `
    <thead><tr><th>knot</th><th>x</th><th>y</th><th>deriv</th></tr></thead>
    <tbody>${xs.map((x, i) => `<tr><td class="label">k${i}</td><td>${x.toFixed(4)}</td><td>${ys[i].toFixed(4)}</td><td>${derivs[i].toFixed(4)}</td></tr>`).join('')}</tbody>
  `;
  root.appendChild(tab);
}

function renderPerCodec(entry, root) {
  const r = decodePerCodecCalibration(entry.value_hex);
  if (r.error) {
    root.innerHTML = `<div class="hint" style="color: var(--bad);">decode error: ${r.error}</div>`;
    return;
  }
  root.innerHTML = `<div class="block-label">${r.entries.length} codecs</div>`;
  // Table.
  const tab = document.createElement('table');
  tab.className = 'stats-table';
  tab.innerHTML = `
    <thead><tr><th>codec</th><th>alpha</th><th>beta</th><th>score@0</th><th>score@50</th><th>score@100</th></tr></thead>
    <tbody>${r.entries.map(e => {
      const s0 = e.alpha + e.beta * 0;
      const s50 = e.alpha + e.beta * 50;
      const s100 = e.alpha + e.beta * 100;
      return `<tr><td class="label">${e.name}</td><td>${e.alpha.toFixed(4)}</td><td>${e.beta.toFixed(4)}</td><td>${s0.toFixed(2)}</td><td>${s50.toFixed(2)}</td><td>${s100.toFixed(2)}</td></tr>`;
    }).join('')}</tbody>
  `;
  root.appendChild(tab);
  // Overlay line plot.
  const canvas = document.createElement('canvas');
  canvas.width = 720;
  canvas.height = 260;
  canvas.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line); margin-top: 12px;';
  root.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const xMin = 0, xMax = 100;
  // y range: cover all codecs' [score@0, score@100] plus identity.
  const allY = r.entries.flatMap(e => [e.alpha, e.alpha + 100 * e.beta]).concat([0, 100]);
  const yLo = Math.min(...allY);
  const yHi = Math.max(...allY);
  drawAxes(ctx, canvas, xMin, xMax, yLo, yHi, 'spline-calibrated score', 'final score');
  const palette = ['#6cb6ff', '#f6a96a', '#7ee787', '#ff7b72', '#d2a8ff', '#9aa3b2'];
  // Identity reference dashed line.
  ctx.strokeStyle = '#444b5a';
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(0, canvas.height - ((0 - yLo) / (yHi - yLo)) * canvas.height);
  ctx.lineTo(canvas.width, canvas.height - ((100 - yLo) / (yHi - yLo)) * canvas.height);
  ctx.stroke();
  ctx.setLineDash([]);
  // Codec lines.
  let labelY = 14;
  for (let i = 0; i < r.entries.length; i++) {
    const e = r.entries[i];
    ctx.strokeStyle = palette[i % palette.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    const y0 = e.alpha;
    const y100 = e.alpha + 100 * e.beta;
    const py0 = canvas.height - ((y0 - yLo) / (yHi - yLo)) * canvas.height;
    const py100 = canvas.height - ((y100 - yLo) / (yHi - yLo)) * canvas.height;
    ctx.moveTo(0, py0);
    ctx.lineTo(canvas.width, py100);
    ctx.stroke();
    ctx.fillStyle = palette[i % palette.length];
    ctx.font = '11px monospace';
    ctx.fillText(e.name, 8, labelY);
    labelY += 14;
  }
}

function renderAlphaHead(entry, root, n_hidden) {
  const r = decodePerSampleAlphaHead(entry.value_hex, n_hidden);
  if (r.error) {
    root.innerHTML = `<div class="hint" style="color: var(--bad);">decode error: ${r.error} (using n_hidden=${n_hidden} from last layer; if wrong, the panel cannot decode this stage)</div>`;
    return;
  }
  root.innerHTML = `<div class="block-label">n_hidden=${r.n_hidden} · rank head + pool head + per-sample sigmoid gate</div>`;
  const arch = document.createElement('div');
  arch.style.cssText = 'font-family: var(--mono); font-size: 11px; background: var(--panel-2); border: 1px solid var(--line); padding: 12px; line-height: 1.6;';
  arch.innerHTML = `
hidden vector h ∈ ℝ^${r.n_hidden}
   │
   ├─ rank head:  rank_score = dot(rank_w[${r.n_hidden}], h) + ${r.rank_b.toFixed(4)}
   │
   ├─ pool head:  reduce(h) → 4 stats → dot(reducer_w[4], stats) + ${r.reducer_b.toFixed(4)}
   │              · p_norm exponent = ${r.p_norm.toFixed(4)}
   │              · reducer_w = [${r.reducer_w.map(v => v.toFixed(4)).join(', ')}]
   │
   └─ alpha gate: α = σ(dot(w_alpha[${r.n_hidden}], h) + ${r.b_alpha.toFixed(4)})
                  output = α · rank_score + (1 - α) · pool_score
  `;
  root.appendChild(arch);
  // Weight histograms for rank_w and w_alpha.
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;';
  root.appendChild(wrap);
  for (const [name, vals] of [['w_alpha (gate)', r.w_alpha], ['rank_w (rank head)', r.rank_w]]) {
    const box = document.createElement('div');
    box.innerHTML = `<div class="block-label">${name} · range [${Math.min(...vals).toFixed(4)}, ${Math.max(...vals).toFixed(4)}]</div>`;
    const c = document.createElement('canvas');
    c.width = 480;
    c.height = 80;
    c.style.cssText = 'background: var(--panel-2); border: 1px solid var(--line);';
    box.appendChild(c);
    drawHistogram(c, vals);
    wrap.appendChild(box);
  }
}

// ===== Helpers =====

function drawAxes(ctx, canvas, xMin, xMax, yMin, yMax, xLabel, yLabel) {
  ctx.fillStyle = '#181b22';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#2a2f3a';
  ctx.lineWidth = 1;
  // X gridlines.
  for (let i = 0; i <= 10; i++) {
    const x = (i / 10) * canvas.width;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  // Y gridlines.
  for (let i = 0; i <= 5; i++) {
    const y = (i / 5) * canvas.height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
  // Axis labels.
  ctx.fillStyle = '#9aa3b2';
  ctx.font = '11px monospace';
  ctx.fillText(`${xLabel} ∈ [${xMin.toFixed(2)}, ${xMax.toFixed(2)}]`, 8, canvas.height - 6);
  ctx.fillText(`${yLabel} ∈ [${yMin.toFixed(2)}, ${yMax.toFixed(2)}]`, 8, 14);
}

function drawHistogram(canvas, vals) {
  const ctx = canvas.getContext('2d');
  const bins = 40;
  const counts = new Array(bins).fill(0);
  const mn = Math.min(...vals);
  const mx = Math.max(...vals);
  const r = mx - mn || 1;
  for (const v of vals) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor((v - mn) / r * bins)));
    counts[b]++;
  }
  const maxC = Math.max(...counts);
  ctx.fillStyle = '#6cb6ff';
  const w = canvas.width / bins;
  for (let i = 0; i < bins; i++) {
    const h = Math.round((counts[i] / maxC) * canvas.height);
    ctx.fillRect(i * w, canvas.height - h, w - 1, h);
  }
  // Zero line marker (if 0 is within the range).
  if (mn < 0 && mx > 0) {
    const zx = ((0 - mn) / r) * canvas.width;
    ctx.strokeStyle = '#f6a96a';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(zx, 0);
    ctx.lineTo(zx, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
