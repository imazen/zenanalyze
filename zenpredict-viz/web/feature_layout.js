// Zensim 228 / 300 / 372-feature schema layout. Maps numeric f0..f_n
// to semantic names `s<scale>.<channel>.<feature_name>` and block
// tags. Names mirror `zensim/src/metric.rs`.
//
// Track B (2026-05-25): when `web/feature_catalog.json` is loaded
// (via `loadFeatureCatalog()`), tooltips and search use the richer
// math-summary + source-ref info from that file. The static layout
// below is the lookup-of-last-resort.

const CHANNELS = ['Y', 'Cb', 'Cr'];
const N_SCALES = 4;

// Per-block feature names. Source of truth: the doc comment table in
// `zensim/src/metric.rs` around `FEATURES_PER_CHANNEL_BASIC` /
// `_WITH_PEAKS` / `_EXTENDED` / `_IW`.
const BASIC = [
  'ssim_mean',
  'ssim_4th',
  'ssim_2nd',
  'art_mean',
  'art_4th',
  'art_2nd',
  'det_mean',
  'det_4th',
  'det_2nd',
  'mse',
  'hf_energy_loss',
  'hf_mag_loss',
  'hf_energy_gain',
];

const PEAKS = ['ssim_max', 'art_max', 'det_max', 'ssim_l8', 'art_l8', 'det_l8'];
const MASKED = [
  'masked_ssim_mean',
  'masked_ssim_4th',
  'masked_ssim_2nd',
  'masked_art_4th',
  'masked_det_4th',
  'masked_mse',
];
const IWPOOL = [
  'iw_ssim_mean',
  'iw_ssim_4th',
  'iw_ssim_2nd',
  'iw_art_4th',
  'iw_det_4th',
  'iw_mse',
];

const N_BASIC = BASIC.length;       // 13
const N_PEAKS = PEAKS.length;       // 6
const N_MASKED = MASKED.length;     // 6
const N_IW = IWPOOL.length;         // 6

const SIZES = {
  228: { layout: [N_BASIC, N_PEAKS],                                names: [BASIC, PEAKS] },
  300: { layout: [N_BASIC, N_PEAKS, N_MASKED],                       names: [BASIC, PEAKS, MASKED] },
  372: { layout: [N_BASIC, N_PEAKS, N_MASKED, N_IW],                 names: [BASIC, PEAKS, MASKED, IWPOOL] },
};

// Loaded asynchronously from web/feature_catalog.json (Track B).
let CATALOG = null;
let CATALOG_LOAD_PROMISE = null;

export function loadFeatureCatalog() {
  if (CATALOG_LOAD_PROMISE) return CATALOG_LOAD_PROMISE;
  CATALOG_LOAD_PROMISE = fetch('./feature_catalog.json')
    .then(resp => {
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return resp.json();
    })
    .then(j => { CATALOG = j; return j; })
    .catch(err => {
      console.warn('feature_catalog.json not found — falling back to static layout:', err.message);
      return null;
    });
  return CATALOG_LOAD_PROMISE;
}

// Return the loaded catalog or null. Synchronous read of cached state.
export function getCatalog() { return CATALOG; }

// Lookup a feature's full entry from the catalog if loaded.
export function catalogEntry(idx, n_features) {
  if (!CATALOG) return null;
  const schema = CATALOG.zensim_schemas?.[String(n_features)];
  if (!schema) return null;
  return schema.features?.[`f${idx}`] || null;
}

// Given a feature index and total feature count, return a label +
// block tag. Prefers the catalog entry when available, falls back to
// the static layout.
export function featureLabel(idx, n_features) {
  const cat = catalogEntry(idx, n_features);
  if (cat) {
    return {
      label: cat.label,
      block: cat.block,
      scale: cat.scale,
      channel: cat.channel,
      feature: cat.feature_name,
      math_summary: cat.math_summary,
      source_ref: cat.source_ref,
    };
  }
  const spec = SIZES[n_features];
  if (!spec) return { label: `f${idx}`, block: 'unknown' };

  const per_channel_total = spec.layout.reduce((a, b) => a + b, 0);
  const per_scale = 3 * per_channel_total;

  if (idx < 0 || idx >= N_SCALES * per_scale) return { label: `f${idx}`, block: 'out_of_range' };

  const scale = Math.floor(idx / per_scale);
  const within_scale = idx % per_scale;
  const channel_idx = Math.floor(within_scale / per_channel_total);
  const within_channel = within_scale % per_channel_total;

  // Walk blocks to find which one this feature lives in.
  let offset = 0;
  for (let b = 0; b < spec.layout.length; b++) {
    const sz = spec.layout[b];
    if (within_channel < offset + sz) {
      const name = spec.names[b][within_channel - offset];
      const blockTag = ['basic', 'peaks', 'masked', 'iw'][b] || `b${b}`;
      return {
        label: `s${scale}.${CHANNELS[channel_idx]}.${name}`,
        block: blockTag,
        scale,
        channel: CHANNELS[channel_idx],
        feature: name,
      };
    }
    offset += sz;
  }
  return { label: `f${idx}`, block: 'unknown' };
}

// Return the 12 (scale, channel) cell labels in column-major order for
// heatmap row labels.
export function cellLabels() {
  const out = [];
  for (let s = 0; s < N_SCALES; s++) {
    for (const c of CHANNELS) out.push(`s${s}.${c}`);
  }
  return out;
}

// Compute per-block summary statistics on an importance vector.
export function blockStats(values, n_features) {
  const spec = SIZES[n_features];
  if (!spec) return [];
  const per_channel_total = spec.layout.reduce((a, b) => a + b, 0);
  const blocks = [];
  let offset = 0;
  const blockTags = ['basic', 'peaks', 'masked', 'iw'];
  for (let b = 0; b < spec.layout.length; b++) {
    const sz = spec.layout[b];
    const indices = [];
    for (let scale = 0; scale < N_SCALES; scale++) {
      for (let c = 0; c < 3; c++) {
        const base = scale * 3 * per_channel_total + c * per_channel_total + offset;
        for (let k = 0; k < sz; k++) indices.push(base + k);
      }
    }
    const vals = indices.map(i => values[i]).filter(v => Number.isFinite(v));
    const sum = vals.reduce((a, b) => a + b, 0);
    const sorted = [...vals].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
    const max = sorted[sorted.length - 1] ?? 0;
    const min = sorted[0] ?? 0;
    blocks.push({
      tag: blockTags[b] || `b${b}`,
      n: vals.length,
      sum,
      mean: sum / vals.length,
      median,
      min,
      max,
    });
    offset += sz;
  }
  const total = blocks.reduce((a, b) => a + b.sum, 0);
  blocks.forEach(b => (b.pct = (100 * b.sum) / total));
  return blocks;
}

// Filter features by a substring search on label/block/feature_name/math_summary.
// `q` is case-insensitive. Returns indices that match.
export function searchFeatures(q, n_features) {
  const query = (q || '').trim().toLowerCase();
  if (!query) return null;  // signal "no filter"
  const out = [];
  for (let i = 0; i < n_features; i++) {
    const info = featureLabel(i, n_features);
    const hay = [info.label, info.block, info.feature || '', info.math_summary || '']
      .join(' ')
      .toLowerCase();
    if (hay.includes(query)) out.push(i);
  }
  return out;
}
