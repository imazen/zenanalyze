// zenpredict-viz entry. Loads WASM, wires UI panels, dispatches bake bytes.

import init, { parse_bake, forward_with_taps, layer_weights } from './pkg/zenpredict_viz.js';
import { renderSummary } from './panels/summary.js';
import { renderScaler } from './panels/scaler.js';
import { renderImportance } from './panels/importance.js';
import { renderForward } from './panels/forward.js';
import { renderWeights } from './panels/weights.js';
import { renderCalibration } from './panels/calibration.js';
import { renderCompare } from './panels/compare.js';
import { renderAttribution } from './panels/attribution.js';
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
  // Loaded asynchronously from web/sample_pack.json. Each entry is
  // { id, label, expected_score, expected_score_source, n_features,
  //   features: [...], source_corpus, ref_basename }. Use the
  // dropdown's selected value (= sample.id) to look up the row.
  samplePack: null,
};

async function bootstrap() {
  await init();
  // Fire-and-forget — the catalog is optional; panels degrade
  // gracefully to the static layout when it's absent.
  loadFeatureCatalog();
  loadSamplePack();
  wireUI();
}

// Fetch the sample_pack sidecar. The file is committed to the repo so
// dev environments serve it from web/; CI builds don't regenerate it.
// Missing sidecar => disable the dropdown and rely on synthetic input.
function loadSamplePack() {
  fetch('./sample_pack.json')
    .then(resp => {
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return resp.json();
    })
    .then(pack => {
      state.samplePack = pack;
      refreshSampleDropdown();
    })
    .catch(err => {
      console.warn('sample_pack.json not found — only synthetic input available:', err.message);
      const sel = document.getElementById('sample-select');
      if (sel) {
        sel.innerHTML = '<option value="">(sample_pack.json missing)</option>';
        sel.disabled = true;
      }
    });
}

function refreshSampleDropdown() {
  const sel = document.getElementById('sample-select');
  if (!sel) return;
  const note = document.getElementById('sample-note');
  if (!state.samplePack || !Array.isArray(state.samplePack.samples)) {
    sel.disabled = true;
    return;
  }
  // Schema gating: hide samples that don't match the current bake's
  // n_inputs. MVP ships only 372-feature samples; show a hint when a
  // 228/300-input bake is loaded.
  const schema = state.samplePack.schema;
  const need = state.summary ? state.summary.n_inputs : null;
  const matches = need === null || need === schema;
  sel.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = matches
    ? '— pick a real (image, quality) sample —'
    : `— no ${need}-feat samples (pack is ${schema}-feat) —`;
  sel.appendChild(placeholder);
  if (!matches) {
    sel.disabled = true;
    if (note) {
      note.style.display = 'block';
      note.textContent =
        `sample_pack ships only ${schema}-input samples; current bake expects ${need}. ` +
        `load v_tuner_v11_2026-05-24 (372-input) to use them.`;
    }
    return;
  }
  sel.disabled = false;
  if (note) { note.style.display = 'none'; note.textContent = ''; }
  for (const s of state.samplePack.samples) {
    const opt = document.createElement('option');
    opt.value = s.id;
    const exp = Number.isFinite(s.expected_score)
      ? ` (≈${s.expected_score.toFixed(1)})`
      : '';
    opt.textContent = `${s.label}${exp}`;
    sel.appendChild(opt);
  }
}

function onSampleSelectChange(e) {
  const id = e.target.value;
  const exp = document.getElementById('sample-expected');
  if (!id || !state.samplePack) {
    if (exp) exp.textContent = '';
    return;
  }
  const sample = state.samplePack.samples.find(s => s.id === id);
  if (!sample) {
    if (exp) exp.textContent = '';
    return;
  }
  const ta = document.getElementById('feature-input');
  if (ta) {
    ta.value = sample.features.map(v => v.toFixed(6)).join(', ');
  }
  if (exp) {
    if (Number.isFinite(sample.expected_score)) {
      exp.innerHTML = `expected ≈ <strong>${sample.expected_score.toFixed(2)}</strong> ` +
        `<span style="color:var(--fg-dim);">(${sample.expected_score_source})</span>`;
    } else {
      exp.innerHTML = `<span style="color:var(--fg-dim);">expected: bake-dependent (zero-vector probe)</span>`;
    }
  }
  // Stash the expected score on state so runForward can show a delta.
  state.lastExpectedScore = Number.isFinite(sample.expected_score) ? sample.expected_score : null;
  state.lastSampleLabel = sample.label;
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
  const sel = document.getElementById('sample-select');
  if (sel) sel.addEventListener('change', onSampleSelectChange);
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
  refreshSampleDropdown();
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
    case 'weights':
      renderWeights(
        state.summary,
        (idx) => layer_weights(state.bakeBytes, idx),
        document.getElementById('weights-body'),
      );
      break;
    case 'calibration':
      renderCalibration(state.summary, document.getElementById('calibration-body'));
      break;
    case 'attribution':
      renderAttribution(
        state.summary,
        (idx) => layer_weights(state.bakeBytes, idx),
        forward_with_taps,
        state.bakeBytes,
        document.getElementById('attribution-body'),
      );
      break;
    case 'compare':
      renderCompare(
        parse_bake,
        (bytes, idx) => layer_weights(bytes, idx),
        document.getElementById('compare-body'),
      );
      break;
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
  renderForward(taps, document.getElementById('forward-body'), state.summary);

  // If a sample is currently picked, show an actual-vs-expected
  // banner. The expected score comes from the corpus row's anchor
  // column; the actual value comes from the bake's final scalar
  // output. They won't match in general — that's the point of the
  // viz, surface the difference.
  if (state.lastExpectedScore != null && taps.output && taps.output.length === 1) {
    const actual = taps.output[0];
    const delta = actual - state.lastExpectedScore;
    const sign = delta >= 0 ? '+' : '';
    const banner = document.createElement('div');
    banner.style.cssText = 'margin-top: 12px; padding: 8px 12px; background: var(--panel-2); ' +
      'border: 1px solid var(--line); border-radius: 4px; font-family: var(--mono); font-size: 11px;';
    banner.innerHTML = `<strong>sample:</strong> ${state.lastSampleLabel}<br>` +
      `<strong>actual MLP output:</strong> ${actual.toFixed(3)} &nbsp; ` +
      `<strong>expected:</strong> ${state.lastExpectedScore.toFixed(3)} &nbsp; ` +
      `<strong>Δ:</strong> <span style="color: ${Math.abs(delta) < 5 ? 'var(--ok)' : 'var(--warn)'}">${sign}${delta.toFixed(3)}</span>` +
      `<div style="color: var(--fg-dim); margin-top: 4px;">expected is the parquet row's anchor column; ` +
      `actual is the raw MLP output (before tanh / spline / per-codec calibration if present).</div>`;
    document.getElementById('forward-body').appendChild(banner);
  }
}

bootstrap();
