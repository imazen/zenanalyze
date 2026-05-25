// Summary panel — bake structural overview + metadata key list.

export function renderSummary(s, root) {
  root.innerHTML = '';
  const layersTable = `
    <h3 style="margin: 20px 0 8px; font-size: 13px;">Layers</h3>
    <table class="stats-table">
      <thead><tr>
        <th>idx</th><th>shape</th><th>dtype</th><th>activation</th>
        <th>bias mean</th><th>bias range</th><th>weight mean</th><th>weight range</th>
      </tr></thead>
      <tbody>${s.layers.map(l => `
        <tr>
          <td class="label">${l.idx}</td>
          <td>${l.in_dim} → ${l.out_dim}</td>
          <td>${l.dtype}</td>
          <td>${l.activation}</td>
          <td>${l.bias_mean.toFixed(4)}</td>
          <td>[${l.bias_min.toFixed(3)}, ${l.bias_max.toFixed(3)}]</td>
          <td>${l.weight_mean.toFixed(4)}</td>
          <td>[${l.weight_min.toFixed(3)}, ${l.weight_max.toFixed(3)}]</td>
        </tr>
      `).join('')}</tbody>
    </table>`;

  const metaTable = s.metadata_keys.length === 0
    ? '<div class="hint">no metadata.</div>'
    : `
    <h3 style="margin: 20px 0 8px; font-size: 13px;">Metadata entries</h3>
    <table class="stats-table">
      <thead><tr><th>key</th><th>kind</th><th>bytes</th></tr></thead>
      <tbody>${s.metadata_keys.map(m => `
        <tr><td class="label">${m.key}</td><td>${m.kind}</td><td>${m.value_len}</td></tr>
      `).join('')}</tbody>
    </table>`;

  root.insertAdjacentHTML('beforeend', layersTable);
  root.insertAdjacentHTML('beforeend', metaTable);
}
