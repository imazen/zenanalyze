// Zensim 372-feature schema layout — maps numeric f0..f371 to semantic names.
// Mirrors `zensim/src/metric.rs` constants FEATURES_PER_CHANNEL_BASIC=13,
// _WITH_PEAKS=19, _EXTENDED=25, _IW=6 across 4 scales × 3 channels.

const CHANNELS = ['Y', 'Cb', 'Cr'];
const N_SCALES = 4;

// Per-block feature names. These mirror the zensim metric.rs source-of-truth
// FeatureView block layouts. If the source ever rotates, regenerate from
// metric.rs.
const BASIC = [
  'ssim_mean',
  'ssim_var',
  'art_mean',
  'art_var',
  'det_mean',
  'det_var',
  'wssim_mean',
  'wssim_var',
  'asym_brt_mean',
  'asym_brt_var',
  'asym_drk_mean',
  'asym_drk_var',
  'csm_mean',
];

const PEAKS = ['ssim_max', 'art_max', 'det_max', 'ssim_p95', 'art_p95', 'det_p95'];
const MASKED = ['mssim_mean', 'mssim_var', 'mart_mean', 'mart_var', 'mdet_mean', 'mdet_var'];
const IWPOOL = ['iw_ssim_p25', 'iw_ssim_p50', 'iw_ssim_p75', 'iw_art_p50', 'iw_det_p50', 'iw_csm_p50'];

const N_BASIC = BASIC.length;       // 13
const N_PEAKS = PEAKS.length;       // 6
const N_MASKED = MASKED.length;     // 6
const N_IW = IWPOOL.length;         // 6

const SIZES = {
  228: { layout: [N_BASIC, N_PEAKS],                                names: [BASIC, PEAKS] },
  300: { layout: [N_BASIC, N_PEAKS, N_MASKED],                       names: [BASIC, PEAKS, MASKED] },
  372: { layout: [N_BASIC, N_PEAKS, N_MASKED, N_IW],                 names: [BASIC, PEAKS, MASKED, IWPOOL] },
};

// Given a feature index and total feature count, return a string label
// `s<scale>.<channel>.<feature_name>` and block tag.
export function featureLabel(idx, n_features) {
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
