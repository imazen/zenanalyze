//! Forward-pass kernel.
//!
//! For each layer:
//! 1. Initialize the accumulator with the layer's biases (broadcast).
//! 2. For each input element `x[i]`, add `x[i] * W[i, :]` to the
//!    accumulator. Embarrassingly parallel across the output dim.
//! 3. Apply activation in-place.
//!
//! The SAXPY-style inner loop is what `magetypes::f32x8` wants when
//! SIMD dispatch lands. The fixed-size `[f32; 8]` chunk loads let
//! LLVM auto-vectorize this to one `f32x8` FMA per iteration on
//! AVX2/AVX-512 and 2× `f32x4` on NEON/WASM today.

use crate::error::PredictError;
use crate::model::{Activation, LEAKY_RELU_ALPHA, LayerView, Model, WeightStorage};

#[cfg(feature = "simd")]
use archmage::autoversion;

/// Run the full forward pass: scale inputs, then layer-by-layer.
///
/// `scratch_a` and `scratch_b` are reused across layers. They must
/// each be at least [`Model::scratch_len`](crate::Model::scratch_len)
/// long. `output` must be exactly `n_outputs` long.
pub fn forward(
    model: &Model,
    features: &[f32],
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
    output: &mut [f32],
) -> Result<(), PredictError> {
    let n_inputs = model.n_inputs();
    let n_outputs = model.n_outputs();
    if features.len() != n_inputs {
        return Err(PredictError::FeatureLenMismatch {
            expected: n_inputs,
            got: features.len(),
        });
    }
    if output.len() != n_outputs {
        return Err(PredictError::FeatureLenMismatch {
            expected: n_outputs,
            got: output.len(),
        });
    }
    let need = model.scratch_len();
    if scratch_a.len() < need || scratch_b.len() < need {
        return Err(PredictError::FeatureLenMismatch {
            expected: need,
            got: scratch_a.len().min(scratch_b.len()),
        });
    }

    // Scale inputs: x' = (x - mean) / scale.
    //
    // Zero-variance columns: sklearn's `_handle_zeros_in_scale`
    // replaces `scale=0` with `1.0` so the column passes through as
    // `(x - mean)`. Mirror that defensively.
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    for i in 0..n_inputs {
        let s = scale[i];
        let safe_s = if s == 0.0 { 1.0 } else { s };
        scratch_a[i] = (features[i] - mean[i]) / safe_s;
    }

    let mut input_buf: &mut [f32] = scratch_a;
    let mut output_buf: &mut [f32] = scratch_b;

    let n_layers = model.n_layers();
    let last_idx = n_layers - 1;

    for (idx, layer) in model.layers().enumerate() {
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;
        let dst: &mut [f32] = if idx == last_idx {
            &mut output[..out_dim]
        } else {
            &mut output_buf[..out_dim]
        };
        let src = &input_buf[..in_dim];
        layer_forward(&layer, src, dst)?;
        if idx != last_idx {
            core::mem::swap(&mut input_buf, &mut output_buf);
        }
    }
    Ok(())
}

fn layer_forward(layer: &LayerView<'_>, src: &[f32], dst: &mut [f32]) -> Result<(), PredictError> {
    let out_dim = layer.out_dim;
    let in_dim = layer.in_dim;
    debug_assert_eq!(src.len(), in_dim);
    debug_assert_eq!(dst.len(), out_dim);
    debug_assert_eq!(layer.biases.len(), out_dim);

    match &layer.weights {
        WeightStorage::F32(w) => {
            dst.copy_from_slice(layer.biases);
            saxpy_matmul_f32(src, w, dst, in_dim, out_dim);
        }
        WeightStorage::F16(w) => {
            dst.copy_from_slice(layer.biases);
            saxpy_matmul_f16(src, w, dst, in_dim, out_dim);
        }
        WeightStorage::I8 { weights, scales } => {
            // Per-output `scales[o]` only applies to the SAXPY
            // accumulator, not the bias. Zero dst, accumulate raw,
            // then `dst[o] = bias[o] + scales[o] * dst[o]`.
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
    Ok(())
}

// The three `saxpy_matmul_*` kernels carry `#[autoversion]` under the
// `simd` feature. It is a **signature-only** transform: archmage re-emits
// the body verbatim as `<name>_v3` under
// `#[target_feature(enable = "avx2,fma")]` plus `<name>_scalar` (the body
// as written), and replaces the original name with a runtime dispatcher.
// Nothing below this comment changes.
//
// Why: baseline x86-64 has no FMA, so the `fma()` helper's `f32::mul_add`
// lowers to an out-of-line software `fmaf` call — 41% of `bake_verdict`'s
// cycles (`perf`, 2026-08-04). Inside the `+fma` variant the same call is
// one `vfmadd` instruction, and LLVM auto-vectorizes the fixed-size
// `[f32; 8]` chunk into `vfmadd231ps`.
//
// Why that is bit-identical, not "close enough":
//   * `f32::mul_add` is IEEE-754 `fusedMultiplyAdd` — a single correctly
//     rounded result. `vfmadd` computes that same operation. Software and
//     hardware FMA agree on every input, including subnormals and NaN
//     payloads. (Rewriting to `a * b + c` would round twice and is
//     deliberately NOT done — see `fma` below.)
//   * The inner loop is a SAXPY, not a reduction: lane `k` accumulates
//     only into `dst[k]`. Widening 8 independent FMAs into one vector FMA
//     reassociates nothing.
// `dispatcher_and_scalar_variant_agree_bitwise` in `simd_parity_tests`
// holds this empirically over random and adversarial inputs.
#[cfg_attr(feature = "simd", autoversion(v3))]
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
                acc_chunk[k] = fma(s, weight_chunk[k], acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                dst[tail_start + k] = fma(s, row[tail_start + k], dst[tail_start + k]);
            }
        }
    }
}

#[cfg_attr(feature = "simd", autoversion(v3))]
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
                acc_chunk[k] = fma(s, wf, acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                let wf = f16_bits_to_f32(row[tail_start + k]);
                dst[tail_start + k] = fma(s, wf, dst[tail_start + k]);
            }
        }
    }
}

#[cfg_attr(feature = "simd", autoversion(v3))]
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
                acc_chunk[k] = fma(s, wf, acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                let wf = row[tail_start + k] as f32;
                dst[tail_start + k] = fma(s, wf, dst[tail_start + k]);
            }
        }
    }
}

/// IEEE-754 binary16 → binary32 converter. Pure integer bit math —
/// works in `no_std` and at compile time. Same answer as
/// `_mm256_cvtph_ps`, one element at a time.
#[inline]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let exp = (h & 0x7c00) >> 10;
    let mant = h & 0x03ff;
    let bits = if exp == 0 {
        if mant == 0 {
            0
        } else {
            // Subnormal — promote to f32 normal.
            let k = 31 - mant.leading_zeros();
            let shift = 10 - k;
            let normalized_mant = (mant << shift) & 0x3ff;
            let f32_exp = k + 103;
            (f32_exp << 23) | (normalized_mant << 13)
        }
    } else if exp == 0x1f {
        0x7f80_0000 | (mant << 13)
    } else {
        ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(sign | bits)
}

/// `a * b + c`. Uses `f32::mul_add` (single-rounding fma) when the
/// `std` feature is on; falls back to `a * b + c` (two roundings)
/// for `no_std + alloc` builds. The numerical difference is in the
/// last bit and well below MLP training noise; the perf difference
/// is one fused instruction vs two.
///
/// **Do not "optimize" the `std` arm to `a * b + c`.** Every shipped
/// zensim score and every baked picker threshold was produced through
/// the single-rounding path; two roundings would move results in the
/// last bit across the whole corpus. The cost of `mul_add` on a CPU
/// without `+fma` enabled is an out-of-line software `fmaf` call — the
/// fix for that is the `simd` feature's `#[autoversion]` on the kernels
/// above (same operation, one instruction), not a change of operation.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "std"))]
    {
        a * b + c
    }
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
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v *= LEAKY_RELU_ALPHA;
                }
            }
        }
    }
}

/// Bit-identity gate for the `simd` feature.
///
/// `#[autoversion]` leaves `<kernel>_scalar` (the body exactly as written,
/// no target features) next to the dispatcher, so the two can be compared
/// directly on the same inputs. Every element of every output must match
/// **by bit pattern** — not by tolerance. A tolerance here would let the
/// two-rounding rewrite this crate forbids slip through unnoticed.
///
/// On a machine without AVX2+FMA the dispatcher selects `_scalar` and the
/// comparison is trivially true; the gate is meaningful on the CI runners
/// and dev boxes that do have it (`archmage::X64V3Token::summon()` reports
/// which case ran).
#[cfg(all(test, feature = "simd"))]
mod simd_parity_tests {
    use super::{saxpy_matmul_f16, saxpy_matmul_f16_scalar};
    use super::{saxpy_matmul_f32, saxpy_matmul_f32_scalar};
    use super::{saxpy_matmul_i8, saxpy_matmul_i8_scalar};
    use archmage::{ScalarToken, SimdToken};

    /// Deterministic xorshift — no rand dependency in the parity gate, so
    /// the exact byte sequence it exercises is reproducible from the seed.
    struct Rng(u64);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        /// Wide dynamic range on purpose: exercises subnormal, huge, and
        /// cancellation-prone magnitudes where a rounding difference
        /// would actually show up.
        fn next_f32(&mut self) -> f32 {
            let m = (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let e = (self.next_u32() % 60) as i32 - 30;
            m * (2.0f32).powi(e)
        }
    }

    /// Shapes chosen to cover both the 8-wide chunk loop and every
    /// possible scalar-tail length (`out_dim % 8` = 0..7), plus the real
    /// model widths zensim ships (944-in / 128-hidden) and a 1×1 edge.
    const SHAPES: &[(usize, usize)] = &[
        (1, 1),
        (3, 7),
        (8, 8),
        (5, 9),
        (16, 13),
        (7, 15),
        (944, 128),
        (128, 1),
        (128, 3),
    ];

    #[test]
    fn dispatcher_and_scalar_variant_agree_bitwise() {
        let tier = if archmage::X64V3Token::summon().is_some() {
            "v3 (AVX2+FMA)"
        } else {
            "scalar (no AVX2+FMA on this host)"
        };
        let st = ScalarToken::summon().expect("ScalarToken is always available");

        let mut rng = Rng(0x5eed_1234_9876_abcd);
        for &(in_dim, out_dim) in SHAPES {
            let src: Vec<f32> = (0..in_dim).map(|_| rng.next_f32()).collect();
            let bias: Vec<f32> = (0..out_dim).map(|_| rng.next_f32()).collect();

            // f32 weights
            let w32: Vec<f32> = (0..in_dim * out_dim).map(|_| rng.next_f32()).collect();
            let mut a = bias.clone();
            let mut b = bias.clone();
            saxpy_matmul_f32(&src, &w32, &mut a, in_dim, out_dim);
            saxpy_matmul_f32_scalar(st, &src, &w32, &mut b, in_dim, out_dim);
            assert_bits(&a, &b, "f32", in_dim, out_dim, tier);

            // f16 weights (raw binary16 bit patterns, full range)
            let w16: Vec<u16> = (0..in_dim * out_dim)
                .map(|_| (rng.next_u32() & 0xffff) as u16)
                .collect();
            let mut a = bias.clone();
            let mut b = bias.clone();
            saxpy_matmul_f16(&src, &w16, &mut a, in_dim, out_dim);
            saxpy_matmul_f16_scalar(st, &src, &w16, &mut b, in_dim, out_dim);
            assert_bits(&a, &b, "f16", in_dim, out_dim, tier);

            // i8 weights
            let w8: Vec<i8> = (0..in_dim * out_dim)
                .map(|_| (rng.next_u32() & 0xff) as u8 as i8)
                .collect();
            let mut a = vec![0.0f32; out_dim];
            let mut b = vec![0.0f32; out_dim];
            saxpy_matmul_i8(&src, &w8, &mut a, in_dim, out_dim);
            saxpy_matmul_i8_scalar(st, &src, &w8, &mut b, in_dim, out_dim);
            assert_bits(&a, &b, "i8", in_dim, out_dim, tier);
        }
    }

    /// Adversarial inputs: zeros (the `s == 0.0` skip branch), infinities,
    /// NaN, and subnormals — the places where a fused-vs-unfused or
    /// reassociated implementation diverges first.
    #[test]
    fn dispatcher_agrees_on_special_values() {
        let tier = if archmage::X64V3Token::summon().is_some() {
            "v3 (AVX2+FMA)"
        } else {
            "scalar"
        };
        let st = ScalarToken::summon().expect("ScalarToken is always available");
        let specials = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE / 4.0, // subnormal
            f32::MAX,
            -f32::MAX,
            1e-30,
            1e30,
        ];
        let in_dim = specials.len();
        let out_dim = 11; // 8-chunk + 3-tail
        let src: Vec<f32> = specials.to_vec();
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| specials[i % specials.len()])
            .collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| specials[i % specials.len()]).collect();

        let mut a = bias.clone();
        let mut b = bias.clone();
        saxpy_matmul_f32(&src, &w, &mut a, in_dim, out_dim);
        saxpy_matmul_f32_scalar(st, &src, &w, &mut b, in_dim, out_dim);
        assert_bits(&a, &b, "f32/special", in_dim, out_dim, tier);
    }

    fn assert_bits(a: &[f32], b: &[f32], what: &str, in_dim: usize, out_dim: usize, tier: &str) {
        assert_eq!(a.len(), b.len());
        for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what} {in_dim}x{out_dim} lane {k}: dispatcher [{tier}] gave {x} \
                 (0x{:08x}), scalar gave {y} (0x{:08x}) — the SIMD variant is NOT \
                 bit-identical",
                x.to_bits(),
                y.to_bits()
            );
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
