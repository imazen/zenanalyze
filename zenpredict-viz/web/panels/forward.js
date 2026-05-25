// Forward panel — waterfall: standardized → per-layer pre/post →
// raw output → [optional] tanh pin → [optional] spline → [optional]
// per-codec affine. Track C extends Stage 1's waterfall with the
// post-MLP calibration stages.

import {
  decodeTanhOutputHead,
  applyTanhPin,
  decodeOutputCalibrationSpline,
  applySpline,
  decodePerCodecCalibration,
} from '../calibration_decoders.js';

export function renderForward(taps, root, summary) {
  root.innerHTML = '';
  const stages = [];
  stages.push({ name: 'standardized', kind: 'input', vals: taps.standardized });
  for (const layer of taps.layer_stages) {
    stages.push({ name: `L${layer.idx} · pre-activation`, vals: layer.pre_activation });
    stages.push({ name: `L${layer.idx} · post-activation`, vals: layer.post_activation });
  }
  stages.push({ name: 'output (raw MLP)', kind: 'output', vals: taps.output });

  // Apply post-MLP calibration stages when present + when the raw output
  // is a single scalar. The per-sample-α head is NOT applied here — it
  // mixes a rank head + pool head from the LAST hidden vector, and is
  // documented in the calibration panel separately.
  if (summary && taps.output.length === 1) {
    const md = new Map((summary.metadata_keys || []).map(m => [m.key, m]));
    let y = taps.output[0];
    const tanh = md.get('zentrain.tanh_output_head');
    if (tanh) {
      const r = decodeTanhOutputHead(tanh.value_hex);
      if (!r.error) {
        y = applyTanhPin(y, r.scale);
        stages.push({ name: `tanh pin (scale=${r.scale.toFixed(3)})`, kind: 'calibrated', vals: [y] });
      }
    }
    const spline = md.get('zentrain.output_calibration_spline');
    if (spline) {
      const r = decodeOutputCalibrationSpline(spline.value_hex);
      if (!r.error) {
        y = applySpline(y, r);
        stages.push({ name: `spline calibration (${r.xs.length} knots)`, kind: 'calibrated', vals: [y] });
      }
    }
    const perCodec = md.get('zentrain.per_codec_calibration');
    if (perCodec) {
      const r = decodePerCodecCalibration(perCodec.value_hex);
      if (!r.error) {
        // We don't know which codec the user is scoring against; show
        // an "identity" (which is what the runtime does for unknown
        // codec hints) plus a per-codec table below.
        stages.push({ name: `per-codec affine (identity — no codec hint)`, kind: 'calibrated', vals: [y], perCodec: r.entries });
      }
    }
    if (md.has('zentrain.per_sample_alpha_head')) {
      stages.push({ name: 'per-sample-α head (see Calibration panel)', kind: 'note', vals: [], note: true });
    }
  }

  const wrap = document.createElement('div');
  wrap.className = 'waterfall';
  let prev = null;
  for (const stage of stages) {
    if (stage.note) {
      const div = document.createElement('div');
      div.className = 'stage';
      div.style.borderLeftColor = 'var(--warn)';
      div.innerHTML = `<span class="name" style="color: var(--warn);">${stage.name}</span><span class="vals" style="color: var(--fg-dim);">multi-head; not a simple scalar transform — see Calibration panel</span>`;
      wrap.appendChild(div);
      continue;
    }
    const vals = Array.from(stage.vals);
    const min = vals.length ? Math.min(...vals) : 0;
    const max = vals.length ? Math.max(...vals) : 0;
    const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    const div = document.createElement('div');
    div.className = `stage ${stage.kind || ''}`;
    let delta = '';
    if (prev !== null && vals.length === prev.length && vals.length > 1) {
      const dist = Math.sqrt(vals.reduce((acc, v, i) => acc + (v - prev[i]) ** 2, 0));
      delta = `<span class="delta">L2Δ=${dist.toFixed(3)}</span>`;
    }
    const valSummary = vals.length === 1
      ? `value = ${vals[0].toFixed(6)}`
      : `dim ${vals.length} · range [${min.toFixed(3)}, ${max.toFixed(3)}] · μ ${mean.toFixed(3)}`;
    div.innerHTML = `
      <span class="name">${stage.name}</span>
      <span class="vals">${valSummary}</span>
      ${delta}
    `;
    wrap.appendChild(div);
    prev = vals;
  }
  root.appendChild(wrap);

  if (taps.output.length === 1) {
    const lastScalar = stages.filter(s => s.vals && s.vals.length === 1).pop();
    if (lastScalar) {
      const calibratedNotice = lastScalar.kind === 'calibrated'
        ? '<span style="color: var(--ok);">(after calibration stages)</span>'
        : '<span style="color: var(--fg-dim);">(no calibration stages applied — raw MLP output)</span>';
      root.insertAdjacentHTML('beforeend', `
        <div style="margin-top: 16px; font-family: var(--mono); font-size: 12px;">
          <strong>final score:</strong> ${lastScalar.vals[0].toFixed(6)} ${calibratedNotice}
        </div>
      `);
    }

    const perCodecStage = stages.find(s => s.perCodec);
    if (perCodecStage) {
      const rawY = perCodecStage.vals[0];
      const rows = perCodecStage.perCodec.map(e => {
        const s = e.alpha + e.beta * rawY;
        return `<tr><td class="label">${e.name}</td><td>α=${e.alpha.toFixed(4)}</td><td>β=${e.beta.toFixed(4)}</td><td>${s.toFixed(3)}</td></tr>`;
      }).join('');
      root.insertAdjacentHTML('beforeend', `
        <h3 style="margin: 16px 0 8px; font-size: 13px;">per-codec calibration on this input</h3>
        <table class="stats-table">
          <thead><tr><th>codec</th><th>alpha</th><th>beta</th><th>score</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      `);
    }
  }
}
