//! Forward-pass kernel.
//!
//! For each layer:
//!   1. Initialize accumulator with biases (broadcast).
//!   2. For each input element x[i], add x[i] * W[i, :] to the
//!      accumulator. This is a SAXPY across all output dims.
//!   3. Apply activation in-place.
//!
//! The SAXPY-style inner loop is what `magetypes::f32x8` wants —
//! each output dim is independent so we vectorize trivially across
//! the output axis. archmage tokens at the outer-layer boundary
//! prevent LLVM from un-inlining the per-tier dispatch.

use crate::model::{Activation, LEAKY_RELU_ALPHA, LayerView, Model, WeightStorage};

/// Run the full forward pass: scale inputs, then layer-by-layer.
///
/// `scratch_a` and `scratch_b` are reused across layers. They must
/// each be at least `max(layer.out_dim, n_inputs)` long. `output`
/// must be exactly `n_outputs` long.
pub fn forward(
    model: &Model<'_>,
    features: &[f32],
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
    output: &mut [f32],
) {
    let n_inputs = model.n_inputs();
    debug_assert_eq!(features.len(), n_inputs);
    debug_assert_eq!(output.len(), model.n_outputs());

    // Scale inputs: x' = (x - mean) / scale.
    //
    // The `scaler_scale` field in the bake stores sklearn's
    // `StandardScaler.scale_` directly — that attribute IS the
    // standard deviation (`np.sqrt(var_)`). sklearn's own
    // `transform` divides by it (`(x - mean) / std`), and the MLP
    // is trained on `scaler.fit_transform(X_tr)` outputs, so this
    // kernel must divide too — anything else feeds the network
    // inputs scaled differently than what it learned from.
    //
    // (Pre-0.1.0 history: an earlier draft of this kernel
    // multiplied by `scale_`, which silently miscalibrated the
    // forward pass relative to training-time evaluation. Fixed
    // before the first publish; no shipped artifacts depend on
    // the multiply convention. Division-by-near-zero isn't a
    // hazard because sklearn refuses to fit StandardScaler when
    // any feature has zero variance — the picker won't have a
    // bake to load in that case.)
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    let cur = &mut scratch_a[..n_inputs];
    for i in 0..n_inputs {
        cur[i] = (features[i] - mean[i]) / scale[i];
    }

    let mut input_buf: &mut [f32] = scratch_a;
    let mut output_buf: &mut [f32] = scratch_b;

    let layers = model.layers();
    let last_idx = layers.len() - 1;

    for (idx, layer) in layers.iter().enumerate() {
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;

        let dst: &mut [f32] = if idx == last_idx {
            &mut output[..out_dim]
        } else {
            &mut output_buf[..out_dim]
        };
        let src = &input_buf[..in_dim];

        layer_forward(layer, src, dst);

        if idx != last_idx {
            // Swap buffers for the next layer.
            core::mem::swap(&mut input_buf, &mut output_buf);
        }
    }
}

/// Compute `dst = activation(W^T * src + b)` for one layer.
fn layer_forward(layer: &LayerView<'_>, src: &[f32], dst: &mut [f32]) {
    let out_dim = layer.out_dim;
    let in_dim = layer.in_dim;
    debug_assert_eq!(src.len(), in_dim);
    debug_assert_eq!(dst.len(), out_dim);
    debug_assert_eq!(layer.biases.len(), out_dim);

    match &layer.weights {
        WeightStorage::F32(w) => {
            // Init accumulator with biases, then add `sum_i src[i] * W[i, :]`.
            dst.copy_from_slice(layer.biases);
            saxpy_matmul_f32(src, w, dst, in_dim, out_dim);
        }
        WeightStorage::F16(w) => {
            dst.copy_from_slice(layer.biases);
            saxpy_matmul_f16(src, w, dst, in_dim, out_dim);
        }
        WeightStorage::I8 { weights, scales } => {
            // i8 path inverts the bias-then-accumulate ordering
            // because the per-output `scales[o]` only applies to the
            // SAXPY accumulator, not the bias. Accumulate raw
            // `sum_i src[i] * (W_i8[i, o] as f32)` into dst, then
            // commit `dst[o] = bias[o] + scales[o] * dst[o]`.
            for v in dst.iter_mut() {
                *v = 0.0;
            }
            saxpy_matmul_i8(src, weights, dst, in_dim, out_dim);
            debug_assert_eq!(scales.len(), out_dim);
            for o in 0..out_dim {
                dst[o] = layer.biases[o] + scales[o] * dst[o];
            }
        }
    }

    apply_activation(dst, layer.activation);
}

/// `dst[o] += sum_i src[i] * W[i, o]` for the f32-stored case.
///
/// Inner loop adds `src[i] * W[i, :]` to `dst[:]` in chunks of 8.
/// Embarrassingly parallel across the output dim. The fixed-size
/// `[f32; 8]` chunk loads let LLVM auto-vectorize this to one f32x8
/// FMA per iteration on AVX2/AVX-512 and 2× f32x4 on NEON/WASM. We
/// can drop in `#[magetypes]` dispatch later for guaranteed
/// vectorization across tiers — for now scalar autovec is plenty
/// fast for the model sizes we ship (15K weights → ~15 µs).
///
/// Layout: weights are input-major, so `W[i, :]` is
/// `&w[i * out_dim..(i + 1) * out_dim]` — contiguous in memory.
fn saxpy_matmul_f32(src: &[f32], w: &[f32], dst: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    let chunks = out_dim / 8;
    let tail = out_dim % 8;

    for i in 0..in_dim {
        let s = src[i];
        if s == 0.0 {
            continue;
        }
        let row = &w[i * out_dim..(i + 1) * out_dim];

        for c in 0..chunks {
            let base = c * 8;
            let weight_chunk: &[f32; 8] = row[base..base + 8].try_into().unwrap();
            let acc_chunk: &mut [f32; 8] = (&mut dst[base..base + 8]).try_into().unwrap();
            for k in 0..8 {
                acc_chunk[k] = s.mul_add(weight_chunk[k], acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                dst[tail_start + k] = s.mul_add(row[tail_start + k], dst[tail_start + k]);
            }
        }
    }
}

/// Same shape as the f32 path, but every weight is stored as a raw
/// IEEE-754 half-precision bit pattern in a `u16` and converted to
/// `f32` per element via [`f16_bits_to_f32`]. The conversion is a
/// handful of bit ops (no FP path); LLVM unrolls the inner 8-wide
/// chunk and on F16C-capable x86 the autovec usually picks
/// `_mm256_cvtph_ps`. Magetypes-dispatched f16↔f32 conversion is the
/// v0.2 work; for now scalar is fast enough at our model sizes.
fn saxpy_matmul_f16(src: &[f32], w: &[u16], dst: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    let chunks = out_dim / 8;
    let tail = out_dim % 8;

    for i in 0..in_dim {
        let s = src[i];
        if s == 0.0 {
            continue;
        }
        let row = &w[i * out_dim..(i + 1) * out_dim];

        for c in 0..chunks {
            let base = c * 8;
            let acc_chunk: &mut [f32; 8] = (&mut dst[base..base + 8]).try_into().unwrap();
            for k in 0..8 {
                let wf = f16_bits_to_f32(row[base + k]);
                acc_chunk[k] = s.mul_add(wf, acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                let wf = f16_bits_to_f32(row[tail_start + k]);
                dst[tail_start + k] = s.mul_add(wf, dst[tail_start + k]);
            }
        }
    }
}

/// `dst[o] += sum_i src[i] * (W_i8[i, o] as f32)` for the i8-stored
/// case. Per-output scaling is applied by the caller after this
/// returns — see `layer_forward`. The on-the-fly `i8 → f32` cast is
/// a single `as` per element; LLVM lowers this to one VPMOVSXBD on
/// AVX2 / VPMOVSXBD on AVX-512 / SXTL on NEON, so the inner chunk is
/// the same shape as the f32 path.
fn saxpy_matmul_i8(src: &[f32], w: &[i8], dst: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    let chunks = out_dim / 8;
    let tail = out_dim % 8;

    for i in 0..in_dim {
        let s = src[i];
        if s == 0.0 {
            continue;
        }
        let row = &w[i * out_dim..(i + 1) * out_dim];

        for c in 0..chunks {
            let base = c * 8;
            let weight_chunk: &[i8; 8] = row[base..base + 8].try_into().unwrap();
            let acc_chunk: &mut [f32; 8] = (&mut dst[base..base + 8]).try_into().unwrap();
            for k in 0..8 {
                let wf = weight_chunk[k] as f32;
                acc_chunk[k] = s.mul_add(wf, acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                let wf = row[tail_start + k] as f32;
                dst[tail_start + k] = s.mul_add(wf, dst[tail_start + k]);
            }
        }
    }
}

/// IEEE-754 half-precision (binary16) → single-precision (binary32)
/// converter. Handles all classes (zero, subnormal, normal, inf, NaN)
/// without relying on any FP intrinsic. Pure integer bit-twiddling so
/// it works in `no_std` and at compile time (when const-fn is
/// stabilized for u32::leading_zeros, which it already is).
///
/// This is what `_mm256_cvtph_ps` does in hardware on F16C — same
/// answer, just one element at a time. Magetypes-dispatched 8-wide
/// version is the v0.2 path.
#[inline]
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let exp = (h & 0x7c00) >> 10;
    let mant = h & 0x03ff;
    let bits = if exp == 0 {
        if mant == 0 {
            // ±0.
            0
        } else {
            // Subnormal — promote to f32 normal.
            let k = 31 - mant.leading_zeros(); // 0..=9 (position of MSB)
            let shift = 10 - k; // 1..=10
            let normalized_mant = (mant << shift) & 0x3ff;
            let f32_exp = k + 103; // (-24 + 127) at k=0, increments with k
            (f32_exp << 23) | (normalized_mant << 13)
        }
    } else if exp == 0x1f {
        // Inf or NaN — propagate. f32 exp=0xff, mantissa shifted left
        // by 13 to fill the wider field. NaN payload preserved (top
        // bits); the quiet bit stays where it was.
        0x7f80_0000 | (mant << 13)
    } else {
        // Normal — rebias exponent, shift mantissa.
        ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(sign | bits)
}

fn apply_activation(buf: &mut [f32], act: Activation) {
    match act {
        Activation::Identity => {}
        Activation::Relu => {
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        Activation::LeakyRelu => {
            // f(x) = x if x >= 0 else alpha * x. Branchless via
            // mul_add of a per-element factor would be marginal here;
            // the simple branch lets LLVM emit a vectorized blend.
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v *= LEAKY_RELU_ALPHA;
                }
            }
        }
    }
}

#[cfg(test)]
mod f16_tests {
    use super::f16_bits_to_f32;

    fn check(bits: u16, expected: f32, what: &str) {
        let got = f16_bits_to_f32(bits);
        if expected.is_nan() {
            assert!(
                got.is_nan(),
                "{what}: bits=0x{bits:04x} expected NaN, got {got}"
            );
        } else {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{what}: bits=0x{bits:04x} got {got} ({:08x}), expected {expected} ({:08x})",
                got.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn zeros_and_signs() {
        check(0x0000, 0.0, "+0");
        check(0x8000, -0.0, "-0");
    }

    #[test]
    fn ones() {
        check(0x3c00, 1.0, "+1.0");
        check(0xbc00, -1.0, "-1.0");
        check(0x4000, 2.0, "+2.0");
        check(0xc000, -2.0, "-2.0");
    }

    #[test]
    fn fractions() {
        // 0.5 = exp=14 (bias-15 → -1), mant=0
        check(0x3800, 0.5, "0.5");
        // 1/3 representable: 0x3555 = 0.333251953125
        check(0x3555, 0.333_251_95, "approx 1/3");
    }

    #[test]
    fn subnormals() {
        // Smallest positive subnormal: 0x0001 = 2^-24 ≈ 5.96e-8
        check(0x0001, 5.960_464_5e-8, "smallest +subnormal");
        // Largest subnormal: 0x03ff
        check(0x03ff, 6.097_555e-5, "largest +subnormal");
        // Negative smallest subnormal
        check(0x8001, -5.960_464_5e-8, "smallest -subnormal");
    }

    #[test]
    fn extremes() {
        // Smallest positive normal: 0x0400 = 2^-14 ≈ 6.10e-5
        check(0x0400, 6.103_515_6e-5, "smallest +normal");
        // Largest normal: 0x7bff = 65504.0
        check(0x7bff, 65504.0, "largest +normal");
    }

    #[test]
    fn inf_nan() {
        check(0x7c00, f32::INFINITY, "+inf");
        check(0xfc00, f32::NEG_INFINITY, "-inf");
        let nan = f16_bits_to_f32(0x7e00);
        assert!(nan.is_nan(), "0x7e00 should be NaN");
    }
}
