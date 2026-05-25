// Feature attribution panel (Track D MVP). For a given input vector,
// computes a cheap per-feature attribution and renders the top
// positive + top negative contributors.
//
// Formula:
//   attribution[i] = standardized[i] · Σ_h |W₀[i, h]|
//
// This is the "input × first-layer-weight-magnitude" approximation:
// it treats every hidden unit as equally important. A real SHAP / IG
// estimator would integrate downstream gradients through the rest of
// the network; that's the stretch follow-up (would need either a
// forward+backward WASM path or a permutation-Shapley sampler).
//
// The current MVP is fast (one matmul-ish pass per evaluation) and
// surfaces the right shape for the user-facing dial: which input
// features pulled the score up vs down vs the bake's training mean.

import { featureLabel } from '../feature_layout.js';

export function renderAttribution(summary, getLayerWeights, forwardWithTaps, bakeBytes, root) {
  root.innerHTML = '';
  const n = summary.n_inputs;

  const intro = document.createElement('div');
  intro.className = 'hint';
  intro.innerHTML = `
    MVP attribution: <code>attribution[i] = standardized[i] · Σ<sub>h</sub> |W₀[i, h]|</code>.
    Shows which input features pulled the activation up vs down vs the bake's
    training mean. SHAP / Integrated-Gradients would account for downstream
    non-linearities; the cheap version is still strongly indicative for
    diagnosing &quot;why did this image score X?&quot; questions.
  `;
  root.appendChild(intro);

  const controls = document.createElement('div');
  controls.className = 'controls';
  controls.innerHTML = `
    <button class="action" id="attr-synth">synthetic input</button>
    <button class="action" id="attr-copy-fwd">copy from forward</button>
    <button class="action" id="attr-run">attribute</button>
  `;
  root.appendChild(controls);

  const ta = document.createElement('textarea');
  ta.rows = 6;
  ta.placeholder = 'paste feature vector here (or click "synthetic"/"copy from forward")...';
  root.appendChild(ta);

  const body = document.createElement('div');
  body.style.marginTop = '16px';
  root.appendChild(body);

  controls.querySelector('#attr-synth').addEventListener('click', () => {
    const vals = [];
    for (let i = 0; i < n; i++) vals.push((Math.sin(0.1 * i) * 5.0).toFixed(4));
    ta.value = vals.join(', ');
  });
  controls.querySelector('#attr-copy-fwd').addEventListener('click', () => {
    const forwardTa = document.getElementById('feature-input');
    if (forwardTa) ta.value = forwardTa.value;
  });
  controls.querySelector('#attr-run').addEventListener('click', () => {
    const tokens = ta.value.trim().split(/[\s,]+/).filter(Boolean);
    if (tokens.length !== n) {
      alert(`expected ${n} features, got ${tokens.length}`);
      return;
    }
    const features = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = parseFloat(tokens[i]);
      if (!Number.isFinite(v)) { alert(`bad value at index ${i}: ${tokens[i]}`); return; }
      features[i] = v;
    }
    let taps;
    try { taps = forwardWithTaps(bakeBytes, features); }
    catch (err) { alert(`forward pass failed: ${err}`); return; }
    let w0;
    try { w0 = getLayerWeights(0); }
    catch (err) { alert(`layer 0 weights failed: ${err}`); return; }
    const l0 = summary.layers[0];
    const sumAbs = new Float64Array(l0.in_dim);
    for (let i = 0; i < l0.in_dim; i++) {
      let s = 0;
      const base = i * l0.out_dim;
      for (let o = 0; o < l0.out_dim; o++) s += Math.abs(w0[base + o]);
      sumAbs[i] = s;
    }
    const standardized = Array.from(taps.standardized);
    const attribution = new Float64Array(n);
    for (let i = 0; i < n; i++) attribution[i] = standardized[i] * sumAbs[i];
    renderResults(body, attribution, standardized, sumAbs, n);
  });
}

function renderResults(root, attribution, standardized, sumAbs, n) {
  root.innerHTML = '';
  const indexed = Array.from(attribution).map((v, i) => ({ v, i, info: featureLabel(i, n) }));
  indexed.sort((a, b) => b.v - a.v);
  const topPos = indexed.slice(0, 20);
  const topNeg = indexed.slice(-20).reverse();
  // Histogram of attributions.
  const allVals = indexed.map(e => e.v);
  const mn = Math.min(...allVals);
  const mx = Math.max(...allVals);
  const sum = allVals.reduce((a, b) => a + b, 0);
  root.insertAdjacentHTML('beforeend', `
    <div class="block-label">total Σ attribution = ${sum.toFixed(3)} · range [${mn.toFixed(3)}, ${mx.toFixed(3)}]</div>
  `);
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr; gap: 24px;';
  wrap.appendChild(makeTable('top positive contributors', topPos, standardized, sumAbs));
  wrap.appendChild(makeTable('top negative contributors', topNeg, standardized, sumAbs));
  root.appendChild(wrap);
}

function makeTable(title, rows, standardized, sumAbs) {
  const box = document.createElement('div');
  box.innerHTML = `
    <h3 style="margin: 0 0 8px; font-size: 13px;">${title}</h3>
    <table class="stats-table">
      <thead><tr><th>#</th><th>idx</th><th>label</th><th>z</th><th>‖W₀[i]‖₁</th><th>attribution</th></tr></thead>
      <tbody>${rows.map((r, j) => {
        const dir = r.v > 0 ? 'var(--ok)' : 'var(--bad)';
        return `<tr title="${(r.info.math_summary || '').replace(/"/g, '&quot;')}"><td class="label">${j + 1}</td><td>f${r.i}</td><td class="label">${r.info.label}</td><td>${standardized[r.i].toFixed(3)}</td><td>${sumAbs[r.i].toFixed(3)}</td><td style="color: ${dir};">${r.v >= 0 ? '+' : ''}${r.v.toFixed(3)}</td></tr>`;
      }).join('')}</tbody>
    </table>
  `;
  return box;
}
