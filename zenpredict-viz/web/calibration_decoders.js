// JS-side TLV decoders for post-MLP zensim calibration stages.
// Wire formats mirror the parse_* helpers in zensim/src/metric.rs.

// Convert a hex string ("ab12...") back to a Uint8Array.
export function hexToBytes(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    out[i / 2] = parseInt(hex.substr(i, 2), 16);
  }
  return out;
}

// Read a little-endian f32 from a Uint8Array.
function readF32LE(bytes, offset) {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  for (let i = 0; i < 4; i++) view.setUint8(i, bytes[offset + i]);
  return view.getFloat32(0, true);
}

function readU32LE(bytes, offset) {
  return (bytes[offset]
       | (bytes[offset + 1] << 8)
       | (bytes[offset + 2] << 16)
       | (bytes[offset + 3] << 24)) >>> 0;
}

// ===== zentrain.tanh_output_head =====
// Payload: single f32 LE (4 bytes) = sigmoid pin scale.
// Runtime: y_score = 100 / (1 + exp(-clamp(y_pre / scale, -30, 30))).
export function decodeTanhOutputHead(hex) {
  const b = hexToBytes(hex);
  if (b.length !== 4) return { error: `bad length: ${b.length}, expected 4` };
  const scale = readF32LE(b, 0);
  if (!Number.isFinite(scale) || scale <= 0) {
    return { error: `bad scale: ${scale} (must be finite, positive)` };
  }
  return { scale };
}

export function applyTanhPin(yPre, scale) {
  const xc = Math.max(-30, Math.min(30, yPre / scale));
  return 100 / (1 + Math.exp(-xc));
}

// ===== zentrain.output_calibration_spline =====
// Payload: [u32 n_knots, n_knots × (f32 x_le, f32 y_le)]
// Total bytes: 4 + 8 * n_knots. xs must be strictly increasing.
export function decodeOutputCalibrationSpline(hex) {
  const b = hexToBytes(hex);
  if (b.length < 4) return { error: `truncated header` };
  const n = readU32LE(b, 0);
  if (n < 2) return { error: `n_knots = ${n} (< 2)` };
  const expected = 4 + 8 * n;
  if (b.length !== expected) return { error: `length ${b.length}, expected ${expected}` };
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const off = 4 + i * 8;
    const x = readF32LE(b, off);
    const y = readF32LE(b, off + 4);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return { error: `non-finite knot at ${i}` };
    xs.push(x);
    ys.push(y);
  }
  for (let i = 1; i < n; i++) {
    if (!(xs[i] > xs[i - 1])) return { error: `xs not strictly increasing at knot ${i}` };
  }
  const derivs = pchipDerivs(xs, ys);
  return { xs, ys, derivs };
}

// Fritsch–Carlson monotone-preserving derivatives (mirror zensim metric.rs).
function pchipDerivs(xs, ys) {
  const n = xs.length;
  if (n === 2) {
    const s = (ys[1] - ys[0]) / (xs[1] - xs[0]);
    return [s, s];
  }
  const h = new Array(n - 1);
  const s = new Array(n - 1);
  for (let k = 0; k < n - 1; k++) {
    h[k] = xs[k + 1] - xs[k];
    s[k] = (ys[k + 1] - ys[k]) / h[k];
  }
  const d = new Array(n).fill(0);
  for (let k = 1; k < n - 1; k++) {
    if (s[k - 1] * s[k] <= 0) {
      d[k] = 0;
    } else {
      const w1 = 2 * h[k] + h[k - 1];
      const w2 = h[k] + 2 * h[k - 1];
      d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k]);
    }
  }
  d[0] = endpoint(h[0], h[1], s[0], s[1]);
  d[n - 1] = endpoint(h[n - 2], h[n - 3], s[n - 2], s[n - 3]);
  return d;
}

function endpoint(h0, h1, s0, s1) {
  const d = ((2 * h0 + h1) * s0 - h0 * s1) / (h0 + h1);
  if (d * s0 <= 0) return 0;
  if (s0 * s1 <= 0 && Math.abs(d) > 3 * Math.abs(s0)) return 3 * s0;
  return d;
}

export function applySpline(x, spline) {
  if (!Number.isFinite(x)) return x;
  const { xs, ys, derivs } = spline;
  const n = xs.length;
  if (x <= xs[0]) return ys[0] + derivs[0] * (x - xs[0]);
  if (x >= xs[n - 1]) return ys[n - 1] + derivs[n - 1] * (x - xs[n - 1]);
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const h = xs[hi] - xs[lo];
  const t = (x - xs[lo]) / h;
  const h00 = (1 + 2 * t) * (1 - t) * (1 - t);
  const h10 = t * (1 - t) * (1 - t);
  const h01 = t * t * (3 - 2 * t);
  const h11 = t * t * (t - 1);
  return h00 * ys[lo] + h10 * h * derivs[lo] + h01 * ys[hi] + h11 * h * derivs[hi];
}

// ===== zentrain.per_codec_calibration =====
// Payload: [u32 n_codecs, n_codecs × (u32 name_len, name_len utf8, f32 alpha, f32 beta)]
// Runtime: score = alpha + beta * raw, beta > 0.
export function decodePerCodecCalibration(hex) {
  const b = hexToBytes(hex);
  if (b.length < 4) return { error: 'truncated header' };
  const n = readU32LE(b, 0);
  let off = 4;
  const entries = [];
  for (let i = 0; i < n; i++) {
    if (off + 4 > b.length) return { error: `truncated at entry ${i} header` };
    const nameLen = readU32LE(b, off);
    off += 4;
    if (off + nameLen + 8 > b.length) return { error: `truncated at entry ${i} body (name_len ${nameLen})` };
    const nameBytes = b.slice(off, off + nameLen);
    let name;
    try {
      name = new TextDecoder('utf-8', { fatal: true }).decode(nameBytes);
    } catch {
      return { error: `non-utf8 name at entry ${i}` };
    }
    off += nameLen;
    const alpha = readF32LE(b, off);
    off += 4;
    const beta = readF32LE(b, off);
    off += 4;
    if (!Number.isFinite(alpha) || !Number.isFinite(beta) || beta <= 0) {
      return { error: `entry ${i} (${name}): bad alpha=${alpha} beta=${beta}` };
    }
    entries.push({ name, alpha, beta });
  }
  return { entries };
}

// ===== zentrain.per_sample_alpha_head =====
// Payload: (2 * n_hidden + 8) f32 LE entries.
// Layout (in order):
//   w_alpha:   n_hidden f32  (alpha logit weights over hidden vector)
//   b_alpha:   1 f32         (alpha logit bias)
//   rank_w:    n_hidden f32  (rank-head weights over hidden vector)
//   rank_b:    1 f32         (rank-head bias)
//   reducer_w: 4 f32         (4-feature mean→pool reducer weights)
//   reducer_b: 1 f32         (reducer bias)
//   p_norm:    1 f32         (pool-head p-norm exponent)
//
// Runtime: see zensim::metric::apply_per_sample_alpha_head. Mixes a
// rank head (dot(rank_w, h) + rank_b) with a pool head computed from
// hidden-vector summary stats, gated by sigmoid(dot(w_alpha, h) + b_alpha).
export function decodePerSampleAlphaHead(hex, nHidden) {
  const b = hexToBytes(hex);
  const expected = (2 * nHidden + 8) * 4;
  if (b.length !== expected) {
    return { error: `length ${b.length}, expected ${expected} (n_hidden=${nHidden})` };
  }
  const f = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const w_alpha = Array.from(f.slice(0, nHidden));
  const b_alpha = f[nHidden];
  const rank_w = Array.from(f.slice(nHidden + 1, 2 * nHidden + 1));
  const rank_b = f[2 * nHidden + 1];
  const reducer_w = [f[2 * nHidden + 2], f[2 * nHidden + 3], f[2 * nHidden + 4], f[2 * nHidden + 5]];
  const reducer_b = f[2 * nHidden + 6];
  const p_norm = f[2 * nHidden + 7];
  return { w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm, n_hidden: nHidden };
}
