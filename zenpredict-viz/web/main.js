// zenpredict-viz entry. Loads WASM, wires UI panels, dispatches bake bytes.

import init, { parse_bake, forward_with_taps } from './pkg/zenpredict_viz.js';
import { renderSummary } from './panels/summary.js';
import { renderScaler } from './panels/scaler.js';
import { renderImportance } from './panels/importance.js';
import { renderForward } from './panels/forward.js';
import { loadFeatureCatalog } from './feature_layout.js';

const KNOWN_STAGES = [
  'zentrain.feature_transforms',
  'zentrain.feature_transform_params',
  'zentrain.tanh_output_head',
  'zentrain.output_calibration_spline',
  'zentrain.per_sample_alpha_head',
  'zentrain.per_codec_calibration',
  'zentrain.hybrid_heads_layout',
];

const state = {
  bakeBytes: null,
  summary: null,
};

async function bootstrap() {
  await init();
  // Fire-and-forget — the catalog is optional; panels degrade
  // gracefully to the static layout when it's absent.
  loadFeatureCatalog();
  wireUI();
}

function wireUI() {
  document.querySelectorAll('nav button').forEach(btn => {
    btn.addEventListener('click', () => switchPanel(btn.dataset.panel));
  });

  const fileInput = document.getElementById('file-input');
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const bytes = new Uint8Array(await file.arrayBuffer());
    await loadBake(bytes, file.name);
  });

  document.getElementById('quick-bakes').addEventListener('click', async (e) => {
    if (e.target.tagName !== 'BUTTON') return;
    const url = e.target.dataset.url;
    const name = e.target.textContent;
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const bytes = new Uint8Array(await resp.arrayBuffer());
      await loadBake(bytes, name);
    } catch (err) {
      alert(`failed to load ${url}: ${err.message}\n\nto use shipped bakes, copy them to web/bakes/ in this directory.`);
    }
  });

  document.getElementById('synth-features').addEventListener('click', synthFeatures);
  document.getElementById('run-forward').addEventListener('click', runForward);
}

async function loadBake(bytes, name) {
  state.bakeBytes = bytes;
  try {
    state.summary = parse_bake(bytes);
  } catch (err) {
    alert(`parse failed: ${err}`);
    state.summary = null;
    return;
  }
  document.getElementById('bake-name').textContent = `${name} · ${bytes.length} bytes`;
  renderSidebar();
  refreshActivePanel();
}

function renderSidebar() {
  const s = state.summary;
  const list = document.getElementById('summary-sidebar');
  if (!s) {
    list.innerHTML = '<div class="empty">load a bake to begin</div>';
    return;
  }
  list.innerHTML = '';
  const rows = [
    ['inputs', s.n_inputs],
    ['outputs', s.n_outputs],
    ['layers', s.n_layers],
    ['schema_hash', `0x${s.schema_hash.toString(16).padStart(16, '0')}`],
    ['bake bytes', s.bake_bytes.toLocaleString()],
  ];
  for (const [k, v] of rows) {
    list.insertAdjacentHTML('beforeend', `<div><span>${k}</span><span class="val">${v}</span></div>`);
  }
  const badges = document.getElementById('stage-badges');
  badges.innerHTML = '';
  const keyset = new Set(s.metadata_keys.map(m => m.key));
  for (const k of KNOWN_STAGES) {
    const present = keyset.has(k);
    const short = k.replace('zentrain.', '');
    badges.insertAdjacentHTML(
      'beforeend',
      `<span class="stage-badge ${present ? 'present' : ''}" title="${k}">${short}</span>`
    );
  }
}

function switchPanel(name) {
  document.querySelectorAll('nav button').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.panel === name);
  });
  document.querySelectorAll('section.panel').forEach(p => {
    p.classList.toggle('active', p.dataset.panel === name);
  });
  refreshActivePanel();
}

function refreshActivePanel() {
  if (!state.summary) return;
  const active = document.querySelector('section.panel.active');
  if (!active) return;
  const name = active.dataset.panel;
  switch (name) {
    case 'summary': renderSummary(state.summary, document.getElementById('summary-body')); break;
    case 'scaler': renderScaler(state.summary, document.getElementById('scaler-body')); break;
    case 'importance': renderImportance(state.summary, document.getElementById('importance-body')); break;
    case 'forward': /* user-driven */ break;
  }
}

function synthFeatures() {
  if (!state.summary) { alert('load a bake first'); return; }
  const n = state.summary.n_inputs;
  const vals = [];
  for (let i = 0; i < n; i++) vals.push((Math.sin(0.1 * i) * 5.0).toFixed(4));
  document.getElementById('feature-input').value = vals.join(', ');
}

async function runForward() {
  if (!state.summary) { alert('load a bake first'); return; }
  const raw = document.getElementById('feature-input').value.trim();
  const tokens = raw.split(/[\s,]+/).filter(Boolean);
  if (tokens.length !== state.summary.n_inputs) {
    alert(`expected ${state.summary.n_inputs} features, got ${tokens.length}`);
    return;
  }
  const features = new Float32Array(tokens.length);
  for (let i = 0; i < tokens.length; i++) {
    const v = parseFloat(tokens[i]);
    if (!Number.isFinite(v)) { alert(`bad value at index ${i}: ${tokens[i]}`); return; }
    features[i] = v;
  }
  let taps;
  try {
    taps = forward_with_taps(state.bakeBytes, features);
  } catch (err) {
    alert(`forward pass failed: ${err}`); return;
  }
  renderForward(taps, document.getElementById('forward-body'));
}

bootstrap();
