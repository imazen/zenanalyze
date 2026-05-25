// Forward panel — waterfall: standardized → per-layer pre/post → output.

export function renderForward(taps, root) {
  root.innerHTML = '';
  const stages = [];
  stages.push({ name: 'standardized', kind: 'input', vals: taps.standardized });
  for (const layer of taps.layer_stages) {
    stages.push({ name: `L${layer.idx} · pre-activation`, vals: layer.pre_activation });
    stages.push({ name: `L${layer.idx} · post-activation`, vals: layer.post_activation });
  }
  stages.push({ name: 'output (raw)', kind: 'output', vals: taps.output });

  const wrap = document.createElement('div');
  wrap.className = 'waterfall';
  let prev = null;
  for (const stage of stages) {
    const vals = Array.from(stage.vals);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const div = document.createElement('div');
    div.className = `stage ${stage.kind || ''}`;
    let delta = '';
    if (prev !== null && vals.length === prev.length) {
      const dist = Math.sqrt(vals.reduce((acc, v, i) => acc + (v - prev[i]) ** 2, 0));
      delta = `<span class="delta">L2Δ=${dist.toFixed(3)}</span>`;
    }
    div.innerHTML = `
      <span class="name">${stage.name}</span>
      <span class="vals">dim ${vals.length} · range [${min.toFixed(3)}, ${max.toFixed(3)}] · μ ${mean.toFixed(3)}</span>
      ${delta}
    `;
    wrap.appendChild(div);
    prev = vals;
  }
  root.appendChild(wrap);

  if (taps.output.length === 1) {
    root.insertAdjacentHTML('beforeend', `
      <div style="margin-top: 16px; font-family: var(--mono); font-size: 12px;">
        <strong>final raw output:</strong> ${taps.output[0].toFixed(6)}
      </div>
      <div class="hint" style="margin-top: 8px;">post-MLP calibration stages (tanh-pin, spline, per-codec) are not applied in this view yet — see issue #79 P1.</div>
    `);
  }
}
