//! Synthetic-model roundtrip tests.
//!
//! Builds a tiny 2-layer MLP in memory, serializes it to the v1
//! binary format, loads it via `Model::from_bytes`, runs inference,
//! and verifies the output matches a hand-computed reference.

use crate::{AllowedMask, Model, Picker};

/// 8-aligned byte buffer for tests. Real callers wrap their
/// `include_bytes!` literal in `#[repr(C, align(8))] struct
/// Aligned([u8; N])` to guarantee alignment.
struct AlignedBuf {
    storage: alloc::boxed::Box<[u64]>,
    len: usize,
}

impl AlignedBuf {
    fn from_slice(src: &[u8]) -> Self {
        let n_u64 = src.len().div_ceil(8);
        let mut storage = alloc::vec![0u64; n_u64.max(1)].into_boxed_slice();
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
        bytes[..src.len()].copy_from_slice(src);
        Self {
            storage,
            len: src.len(),
        }
    }

    fn as_bytes(&self) -> &[u8] {
        let bytes: &[u8] = bytemuck::cast_slice(&self.storage);
        &bytes[..self.len]
    }
}

/// Convert an f32 to its IEEE-754 half-precision bit pattern for
/// test bake. Matches the inference-side `f16_bits_to_f32`. Same
/// algorithm shape: handle zero/subnormal/inf/NaN/normal cases by
/// pure bit math. Round-to-nearest-even on the mantissa truncation.
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let mant32 = bits & 0x007f_ffff;
    if exp32 == 0xff {
        // f32 inf/NaN
        let m = if mant32 != 0 {
            ((mant32 >> 13) | 0x200) as u16
        } else {
            0
        };
        (sign << 15) | 0x7c00 | m
    } else if exp32 == 0 {
        // f32 zero/subnormal — flushes to ±0 in f16 (subnormals too small).
        sign << 15
    } else {
        let exp16 = exp32 - 127 + 15;
        if exp16 >= 0x1f {
            // Overflow → inf
            (sign << 15) | 0x7c00
        } else if exp16 <= 0 {
            // Underflow into subnormal or zero
            if exp16 < -10 {
                sign << 15
            } else {
                let mant_with_implicit = mant32 | 0x0080_0000; // bit 23 = implicit 1
                let shift = 14 - exp16; // shift to land in 10-bit f16 mantissa
                let m = mant_with_implicit >> shift;
                // Round-to-nearest-even
                let half = 1u32 << (shift - 1);
                let m = if (mant_with_implicit & half) != 0
                    && ((mant_with_implicit & (half - 1)) != 0 || (m & 1) != 0)
                {
                    m + 1
                } else {
                    m
                };
                (sign << 15) | (m as u16)
            }
        } else {
            // Normal, possibly with overflow on round
            let m = mant32 >> 13;
            // Round-to-nearest-even
            let half = 1u32 << 12;
            let lower = mant32 & 0x1fff;
            let m = if lower > half || (lower == half && (m & 1) != 0) {
                m + 1
            } else {
                m
            };
            // Carry-out from mantissa rounds exp
            let mut e = exp16 as u32;
            let mut m = m;
            if m & 0x400 != 0 {
                m = 0;
                e += 1;
            }
            if e >= 0x1f {
                (sign << 15) | 0x7c00
            } else {
                (sign << 15) | ((e as u16) << 10) | (m as u16)
            }
        }
    }
}

/// Test layer-spec tuple: (in_dim, out_dim, activation_byte, weights, biases).
type LayerSpec<'a> = (usize, usize, u8, &'a [f32], &'a [f32]);

/// Per-output i8 quantize a row-major f32 weight block of shape
/// (in_dim, out_dim). Mirrors `tools/bake_picker.py`'s scheme:
/// `scale[o] = max_i |W[i, o]| / 127`, `q[i, o] = round(W / scale)`.
/// All-zero columns get `scale = 1.0` to avoid div-by-zero.
fn quantize_i8_per_output(
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> (alloc::vec::Vec<i8>, alloc::vec::Vec<f32>) {
    let mut scales = alloc::vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut m = 0.0f32;
        for i in 0..in_dim {
            let a = w[i * out_dim + o].abs();
            if a > m {
                m = a;
            }
        }
        scales[o] = if m == 0.0 { 1.0 } else { m / 127.0 };
    }
    let mut q = alloc::vec![0i8; in_dim * out_dim];
    for i in 0..in_dim {
        for o in 0..out_dim {
            let v = w[i * out_dim + o] / scales[o];
            let r = v.round().clamp(-128.0, 127.0) as i32;
            q[i * out_dim + o] = r as i8;
        }
    }
    (q, scales)
}

/// Build a v1 model with i8 weights into `out`. Per-output f32
/// scales follow the i8 weight block; biases stay f32. Identity
/// scaler.
fn write_v1_model_i8(
    out: &mut alloc::vec::Vec<u8>,
    n_inputs: usize,
    layers: &[LayerSpec<'_>],
    schema_hash: u64,
) {
    let n_outputs = layers.last().unwrap().1;
    let n_layers = layers.len();
    out.clear();
    out.extend_from_slice(b"ZNPK");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&(n_inputs as u32).to_le_bytes());
    out.extend_from_slice(&(n_outputs as u32).to_le_bytes());
    out.extend_from_slice(&(n_layers as u32).to_le_bytes());
    out.extend_from_slice(&schema_hash.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), 32);
    for _ in 0..n_inputs {
        out.extend_from_slice(&0.0f32.to_le_bytes());
    }
    for _ in 0..n_inputs {
        out.extend_from_slice(&1.0f32.to_le_bytes());
    }
    for &(in_d, out_d, act, w, b) in layers {
        out.extend_from_slice(&(in_d as u32).to_le_bytes());
        out.extend_from_slice(&(out_d as u32).to_le_bytes());
        out.push(act);
        out.push(2); // weight_dtype = i8
        out.extend_from_slice(&[0, 0]);
        let (q, scales) = quantize_i8_per_output(w, in_d, out_d);
        for &v in &q {
            out.push(v as u8);
        }
        // Pad i8 block to 4-byte alignment for the f32 scales/biases.
        let pad = (4 - ((in_d * out_d) % 4)) % 4;
        for _ in 0..pad {
            out.push(0);
        }
        for &s in &scales {
            out.extend_from_slice(&s.to_le_bytes());
        }
        for &val in b {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
}

/// Build a v1 model with f16 weights into `out`. f32 biases.
fn write_v1_model_f16(
    out: &mut alloc::vec::Vec<u8>,
    n_inputs: usize,
    layers: &[LayerSpec<'_>],
    schema_hash: u64,
) {
    let n_outputs = layers.last().unwrap().1;
    let n_layers = layers.len();
    out.clear();
    out.extend_from_slice(b"ZNPK");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&(n_inputs as u32).to_le_bytes());
    out.extend_from_slice(&(n_outputs as u32).to_le_bytes());
    out.extend_from_slice(&(n_layers as u32).to_le_bytes());
    out.extend_from_slice(&schema_hash.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), 32);
    for _ in 0..n_inputs {
        out.extend_from_slice(&0.0f32.to_le_bytes());
    }
    for _ in 0..n_inputs {
        out.extend_from_slice(&1.0f32.to_le_bytes());
    }
    for &(in_d, out_d, act, w, b) in layers {
        out.extend_from_slice(&(in_d as u32).to_le_bytes());
        out.extend_from_slice(&(out_d as u32).to_le_bytes());
        out.push(act);
        out.push(1); // weight_dtype = f16
        out.extend_from_slice(&[0, 0]);
        for &val in w {
            out.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes());
        }
        // Pad to 4-aligned for f32 biases when n_weights is odd.
        if (in_d * out_d) % 2 == 1 {
            out.extend_from_slice(&[0, 0]);
        }
        for &val in b {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
}

/// Build a v1 model with f32 weights into `out`.
///
/// `mean[i] = 0.0`, `scale[i] = 1.0` (identity scaler) for tests.
fn write_v1_model_f32(
    out: &mut alloc::vec::Vec<u8>,
    n_inputs: usize,
    layers: &[LayerSpec<'_>],
    schema_hash: u64,
) {
    write_v1_model_f32_with_scaler(
        out,
        n_inputs,
        &alloc::vec![0.0; n_inputs],
        &alloc::vec![1.0; n_inputs],
        layers,
        schema_hash,
    );
}

/// Variant that takes explicit `(mean, scale)` for the input scaler
/// instead of the default identity (mean=0, scale=1). Lets tests
/// exercise the standardize step with realistic non-trivial values
/// — important for the `(x - mean) / scale` regression — and is
/// also handy for synthesizing models that mimic
/// `train_hybrid.py`'s emit shape.
fn write_v1_model_f32_with_scaler(
    out: &mut alloc::vec::Vec<u8>,
    n_inputs: usize,
    scaler_mean: &[f32],
    scaler_scale: &[f32],
    layers: &[LayerSpec<'_>],
    schema_hash: u64,
) {
    debug_assert_eq!(scaler_mean.len(), n_inputs);
    debug_assert_eq!(scaler_scale.len(), n_inputs);
    let n_outputs = layers.last().unwrap().1;
    let n_layers = layers.len();
    out.clear();
    out.extend_from_slice(b"ZNPK");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&(n_inputs as u32).to_le_bytes());
    out.extend_from_slice(&(n_outputs as u32).to_le_bytes());
    out.extend_from_slice(&(n_layers as u32).to_le_bytes());
    out.extend_from_slice(&schema_hash.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), 32);
    for &m in scaler_mean {
        out.extend_from_slice(&m.to_le_bytes());
    }
    for &s in scaler_scale {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for &(in_d, out_d, act, w, b) in layers {
        out.extend_from_slice(&(in_d as u32).to_le_bytes());
        out.extend_from_slice(&(out_d as u32).to_le_bytes());
        out.push(act);
        out.push(0);
        out.extend_from_slice(&[0, 0]);
        for &val in w {
            out.extend_from_slice(&val.to_le_bytes());
        }
        for &val in b {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
}

#[test]
fn parse_minimal_one_layer_identity() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(
        &mut buf,
        2,
        &[(2, 3, 0, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0.0, 0.0, 1.0])],
        0xDEAD_BEEF_CAFE_F00D,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    assert_eq!(model.n_inputs(), 2);
    assert_eq!(model.n_outputs(), 3);
    assert_eq!(model.schema_hash(), 0xDEAD_BEEF_CAFE_F00D);

    let mut picker = Picker::new(model);
    let out = picker.predict(&[3.0, 5.0]).unwrap();
    assert_eq!(out, &[3.0, 5.0, 1.0]);
}

#[test]
fn leaky_relu_scales_negatives() {
    // Layer: 1 → 2 with W = [-2.0, 1.0], b = [0.0, 0.0]. Activation
    // byte 2 = LeakyRelu(0.01). Input 1.0 produces pre-activations
    // [-2.0, 1.0], so post-activation should be [-0.02, 1.0].
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(&mut buf, 1, &[(1, 2, 2, &[-2.0, 1.0], &[0.0, 0.0])], 0);
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut picker = Picker::new(model);
    let out = picker.predict(&[1.0]).unwrap();
    assert!(
        (out[0] - (-0.02)).abs() < 1e-6,
        "leaky relu negative leg: got {} expected -0.02",
        out[0]
    );
    assert!((out[1] - 1.0).abs() < 1e-6);
}

#[test]
fn relu_zeros_negatives() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(&mut buf, 1, &[(1, 2, 1, &[-2.0, 1.0], &[0.0, 0.0])], 0);
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut picker = Picker::new(model);
    let out = picker.predict(&[1.0]).unwrap();
    assert_eq!(out, &[0.0, 1.0]);
}

#[test]
fn two_layer_mlp() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(
        &mut buf,
        2,
        &[
            (
                2,
                4,
                1,
                &[1.0, -1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 1.0],
                &[0.0, 0.0, 0.0, 0.0],
            ),
            (
                4,
                3,
                0,
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                &[0.0, 0.0, 0.0],
            ),
        ],
        0,
    );

    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut picker = Picker::new(model);
    let out = picker.predict(&[2.0, 3.0]).unwrap();
    assert!((out[0] - 5.0).abs() < 1e-5);
    assert!((out[1] - 4.0).abs() < 1e-5);
    assert!((out[2] - 5.5).abs() < 1e-5);
}

#[test]
fn argmin_picks_smallest_allowed() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(
        &mut buf,
        1,
        &[(1, 4, 0, &[0.0, 0.0, 0.0, 0.0], &[3.0, 1.0, 2.0, 0.5])],
        0,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut picker = Picker::new(model);

    let mask_all = AllowedMask::new(&[true, true, true, true]);
    assert_eq!(
        picker.argmin_masked(&[0.0], &mask_all, None).unwrap(),
        Some(3)
    );

    let mask_no3 = AllowedMask::new(&[true, true, true, false]);
    assert_eq!(
        picker.argmin_masked(&[0.0], &mask_no3, None).unwrap(),
        Some(1)
    );

    let mask_none = AllowedMask::new(&[false, false, false, false]);
    assert_eq!(
        picker.argmin_masked(&[0.0], &mask_none, None).unwrap(),
        None
    );
}

#[test]
fn top_k_picks_smallest_allowed_in_order() {
    let mut buf = alloc::vec::Vec::new();
    // Predictions: [3.0, 1.0, 2.0, 0.5]
    write_v1_model_f32(
        &mut buf,
        1,
        &[(1, 4, 0, &[0.0, 0.0, 0.0, 0.0], &[3.0, 1.0, 2.0, 0.5])],
        0,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut picker = Picker::new(model);

    let mask_all = AllowedMask::new(&[true, true, true, true]);
    let top2 = picker
        .argmin_masked_top_k::<2>(&[0.0], &mask_all, None)
        .unwrap();
    assert_eq!(top2, [Some(3), Some(1)]);

    let top3 = picker
        .argmin_masked_top_k::<3>(&[0.0], &mask_all, None)
        .unwrap();
    assert_eq!(top3, [Some(3), Some(1), Some(2)]);

    let top4 = picker
        .argmin_masked_top_k::<4>(&[0.0], &mask_all, None)
        .unwrap();
    assert_eq!(top4, [Some(3), Some(1), Some(2), Some(0)]);

    // K larger than number of allowed entries — surplus slots are None.
    let top5 = picker
        .argmin_masked_top_k::<5>(&[0.0], &mask_all, None)
        .unwrap();
    assert_eq!(top5, [Some(3), Some(1), Some(2), Some(0), None]);

    // Mask out the best.
    let mask_no3 = AllowedMask::new(&[true, true, true, false]);
    let top2_masked = picker
        .argmin_masked_top_k::<2>(&[0.0], &mask_no3, None)
        .unwrap();
    assert_eq!(top2_masked, [Some(1), Some(2)]);

    // No allowed entries — all None.
    let mask_none = AllowedMask::new(&[false, false, false, false]);
    let top2_empty = picker
        .argmin_masked_top_k::<2>(&[0.0], &mask_none, None)
        .unwrap();
    assert_eq!(top2_empty, [None, None]);
}

#[test]
fn top_k_in_range_returns_subrange_indices() {
    let mut buf = alloc::vec::Vec::new();
    // 6 outputs split notionally [bytes_log[0..3], scalar1[0..3]].
    write_v1_model_f32(
        &mut buf,
        1,
        &[(
            1,
            6,
            0,
            &[0.0; 6],
            &[3.0, 1.0, 2.0, /* scalar tail */ 999.0, 999.0, 999.0],
        )],
        0,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut picker = Picker::new(model);

    let mask_all = AllowedMask::new(&[true, true, true]);
    let top2 = picker
        .argmin_masked_top_k_in_range::<2>(&[0.0], (0, 3), &mask_all, None)
        .unwrap();
    // Indices are within the sub-range, so 1 (= 1.0) and 2 (= 2.0).
    assert_eq!(top2, [Some(1), Some(2)]);
}

#[test]
fn reach_gate_mask_thresholds_correctly() {
    use crate::reach_gate_mask;

    // Realistic per-cell reach rates at some target_zq band.
    let rates = [1.0_f32, 0.99, 0.98, 0.96, 0.5, 0.0, f32::NAN];
    let mut out = [false; 7];

    // Strict default — only cells at/above 0.99 pass.
    reach_gate_mask(&rates, 0.99, &mut out);
    assert_eq!(out, [true, true, false, false, false, false, false]);

    // Relaxed — 0.95 lets cells 0..3 through.
    reach_gate_mask(&rates, 0.95, &mut out);
    assert_eq!(out, [true, true, true, true, false, false, false]);

    // Max-quality / disabled — any positive non-NaN cell passes.
    reach_gate_mask(&rates, 0.0, &mut out);
    assert_eq!(out, [true, true, true, true, true, true, false]);

    // NaN never passes regardless of threshold.
    reach_gate_mask(&[f32::NAN; 3], 0.0, &mut out[..3]);
    assert_eq!(&out[..3], &[false, false, false]);
}

#[test]
fn scaler_divides_by_std_not_multiplies() {
    // Regression test for the "divide vs multiply" runtime kernel
    // bug fixed before 0.1.0. The bake stores sklearn's
    // `StandardScaler.scale_` directly (= std). The Rust runtime
    // must DIVIDE by scale to match the sklearn-trained MLP's
    // expected input — a multiply silently miscalibrates the
    // forward pass.
    //
    // Synthesize a 1-input, identity 1×1 MLP with mean=10, scale=4.
    // For input x=14, the correctly-standardized value is
    // (14 - 10) / 4 = 1.0. A multiply would give (14 - 10) * 4 = 16.0
    // — the values differ by 16×, more than enough that any future
    // accidental flip back to multiply will fail this assertion
    // loudly.
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32_with_scaler(
        &mut buf,
        1,
        &[10.0],
        &[4.0],
        &[(1, 1, 0, &[1.0], &[0.0])], // y = 1 * x'  (identity)
        0,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut picker = Picker::new(model);

    let out = picker.predict(&[14.0_f32]).unwrap();
    assert_eq!(out.len(), 1);
    let want = 1.0_f32; // (14 - 10) / 4
    assert!(
        (out[0] - want).abs() < 1e-6,
        "scaler kernel produced {} for x=14 (mean=10, scale=4); \
         expected {} = (x - mean) / scale. If you see ~16.0 here, \
         the runtime regressed back to multiply-by-std — see \
         inference.rs comment for full context.",
        out[0],
        want,
    );
}

#[test]
fn first_out_of_distribution_finds_first_violator() {
    use crate::{FeatureBounds, first_out_of_distribution};

    let bounds = [
        FeatureBounds::new(0.0, 1.0),
        FeatureBounds::new(-1.0, 1.0),
        FeatureBounds::new(0.5, 1.5),
    ];
    assert_eq!(first_out_of_distribution(&[0.5, 0.0, 1.0], &bounds), None);
    assert_eq!(first_out_of_distribution(&[0.0, -1.0, 1.5], &bounds), None);
    assert_eq!(
        first_out_of_distribution(&[0.5, -2.0, 1.0], &bounds),
        Some(1)
    );
    assert_eq!(
        first_out_of_distribution(&[2.0, 0.0, 100.0], &bounds),
        Some(0)
    );
    assert_eq!(
        first_out_of_distribution(&[f32::NAN, 0.0, 1.0], &bounds),
        Some(0)
    );
    assert_eq!(
        first_out_of_distribution(&[0.5, f32::INFINITY, 1.0], &bounds),
        Some(1)
    );
}

#[test]
fn pick_with_confidence_returns_gap() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(
        &mut buf,
        1,
        &[(1, 4, 0, &[0.0, 0.0, 0.0, 0.0], &[3.0, 1.0, 2.0, 0.5])],
        0,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut picker = Picker::new(model);

    let mask_all = AllowedMask::new(&[true, true, true, true]);
    let (best, gap) = picker
        .pick_with_confidence(&[0.0], &mask_all, None)
        .unwrap()
        .expect("non-empty mask");
    assert_eq!(best, 3);
    assert!((gap - 0.5).abs() < 1e-5, "gap={gap}");

    let mask_one = AllowedMask::new(&[false, false, false, true]);
    let (best, gap) = picker
        .pick_with_confidence(&[0.0], &mask_one, None)
        .unwrap()
        .expect("one allowed entry");
    assert_eq!(best, 3);
    assert!(gap.is_infinite() && gap > 0.0);

    let mask_none = AllowedMask::new(&[false, false, false, false]);
    assert!(
        picker
            .pick_with_confidence(&[0.0], &mask_none, None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn rejects_bad_magic() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(&mut buf, 1, &[(1, 1, 0, &[1.0], &[0.0])], 0);
    buf[0] = b'X';
    let aligned = AlignedBuf::from_slice(&buf);
    let err = Model::from_bytes(aligned.as_bytes()).unwrap_err();
    assert!(matches!(err, crate::PickerError::BadMagic { .. }));
}

#[test]
fn rejects_bad_layer_dims() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(
        &mut buf,
        2,
        &[
            (2, 4, 0, &[0.0; 8], &[0.0; 4]),
            (3, 1, 0, &[0.0; 3], &[0.0]),
        ],
        0,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let err = Model::from_bytes(aligned.as_bytes()).unwrap_err();
    assert!(matches!(
        err,
        crate::PickerError::LayerDimMismatch { layer: 1, .. }
    ));
}

#[test]
fn rejects_truncated_after_magic() {
    // Valid magic, then runs out before version+header_size are read.
    let buf: alloc::vec::Vec<u8> = b"ZNPK\x01".to_vec();
    let aligned = AlignedBuf::from_slice(&buf);
    let err = Model::from_bytes(aligned.as_bytes()).unwrap_err();
    assert!(
        matches!(err, crate::PickerError::Truncated { .. }),
        "expected Truncated, got {err:?}"
    );
}

#[test]
fn rejects_truncated_layer_weights() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(&mut buf, 1, &[(1, 4, 0, &[1.0; 4], &[0.0; 4])], 0);
    // Drop the last 8 bytes — chops part of the bias section.
    buf.truncate(buf.len() - 8);
    let aligned = AlignedBuf::from_slice(&buf);
    let err = Model::from_bytes(aligned.as_bytes()).unwrap_err();
    assert!(
        matches!(err, crate::PickerError::Truncated { .. }),
        "expected Truncated, got {err:?}"
    );
}

#[test]
fn f16_weights_match_f32_baseline() {
    // Same model, two encodings: f32 and f16. Predictions should
    // match within f16-quantization budget. Pick weights and inputs
    // that f16 can represent exactly so we get bit-for-bit match.
    let weights = [1.0f32, -2.0, 0.5, 0.25];
    let biases = [1.0f32, -1.0];
    let mut buf32 = alloc::vec::Vec::new();
    write_v1_model_f32(&mut buf32, 2, &[(2, 2, 0, &weights, &biases)], 0);
    let mut buf16 = alloc::vec::Vec::new();
    write_v1_model_f16(&mut buf16, 2, &[(2, 2, 0, &weights, &biases)], 0);
    assert!(
        buf16.len() < buf32.len(),
        "f16 encoding should be smaller: {} vs {}",
        buf16.len(),
        buf32.len(),
    );

    let aligned32 = AlignedBuf::from_slice(&buf32);
    let aligned16 = AlignedBuf::from_slice(&buf16);
    let m32 = Model::from_bytes(aligned32.as_bytes()).unwrap();
    let m16 = Model::from_bytes(aligned16.as_bytes()).unwrap();
    let mut p32 = Picker::new(m32);
    let mut p16 = Picker::new(m16);
    let inputs = [3.0f32, 4.0];
    let o32 = p32.predict(&inputs).unwrap().to_vec();
    let o16 = p16.predict(&inputs).unwrap().to_vec();
    assert_eq!(o32.len(), 2);
    assert_eq!(o16.len(), 2);
    for (a, b) in o32.iter().zip(o16.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "f32 vs f16 mismatch: {a} vs {b} (diff {})",
            (a - b).abs()
        );
    }
}

#[test]
fn i8_weight_dtype_round_trips() {
    // Asymmetric per-output scales: column 0 has max-abs 8, column 1
    // has max-abs 0.5. Per-output i8 quantization gives column 0 a
    // resolution of 8/127 ≈ 0.063 and column 1 a resolution of
    // 0.5/127 ≈ 0.0039 — exact behavior at quantization grid points.
    let mut buf = alloc::vec::Vec::new();
    let weights = [
        // in_dim=2, out_dim=2 (input-major):
        // input 0 contributes (8.0, 0.5)
        // input 1 contributes (-4.0, -0.25)
        8.0, 0.5, -4.0, -0.25,
    ];
    write_v1_model_i8(
        &mut buf,
        2,
        &[(2, 2, 0, &weights, &[1.0, -0.5])],
        0x00C0_FFEE_DEAD_BEEF_u64,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    assert!(matches!(
        model.layers()[0].weights,
        crate::model::WeightStorage::I8 { .. }
    ));
    let mut picker = Picker::new(model);
    // Hand-compute: q_8 = round(8 / (8/127)) = 127, q_-4 = -64,
    // dequant col 0 = 127*(8/127) = 8.0, -64*(8/127) ≈ -4.031.
    // q_0.5 = round(0.5 / (0.5/127)) = 127, q_-0.25 = -64,
    // dequant col 1 = 127*(0.5/127) = 0.5, -64*(0.5/127) ≈ -0.252.
    // input (1, 1): out[0] = 1.0 + (8 + -4.031) = 4.969, out[1] = -0.5 + (0.5 + -0.252) = -0.252.
    let out = picker.predict(&[1.0, 1.0]).unwrap();
    assert!(
        (out[0] - 4.969).abs() < 0.01,
        "i8 col0 out: got {} expected ~4.969",
        out[0]
    );
    assert!(
        (out[1] - (-0.252)).abs() < 0.01,
        "i8 col1 out: got {} expected ~-0.252",
        out[1]
    );
}

#[test]
fn f32_vs_i8_within_tolerance() {
    // Compare full f32 forward pass against i8-quantized round-trip
    // on a 4×8×4 ReLU MLP with random-ish weights. Mean abs error
    // should sit well under 1% of typical output magnitudes thanks
    // to per-output scaling.
    let w0: alloc::vec::Vec<f32> = (0..32)
        .map(|i| ((i * 7 + 3) % 17) as f32 * 0.1 - 0.8)
        .collect();
    let b0 = alloc::vec![0.0f32; 8];
    let w1: alloc::vec::Vec<f32> = (0..32)
        .map(|i| ((i * 11 + 5) % 13) as f32 * 0.2 - 1.2)
        .collect();
    let b1 = alloc::vec![0.1f32; 4];

    let mut buf32 = alloc::vec::Vec::new();
    write_v1_model_f32(
        &mut buf32,
        4,
        &[(4, 8, 1, &w0, &b0), (8, 4, 0, &w1, &b1)],
        0,
    );
    let mut bufi8 = alloc::vec::Vec::new();
    write_v1_model_i8(
        &mut bufi8,
        4,
        &[(4, 8, 1, &w0, &b0), (8, 4, 0, &w1, &b1)],
        0,
    );

    let a32 = AlignedBuf::from_slice(&buf32);
    let ai8 = AlignedBuf::from_slice(&bufi8);
    let mut p32 = Picker::new(Model::from_bytes(a32.as_bytes()).unwrap());
    let mut pi8 = Picker::new(Model::from_bytes(ai8.as_bytes()).unwrap());

    let inputs = [0.7f32, -1.2, 0.3, 1.4];
    let o32 = p32.predict(&inputs).unwrap().to_vec();
    let oi8 = pi8.predict(&inputs).unwrap().to_vec();

    for (i, (a, b)) in o32.iter().zip(oi8.iter()).enumerate() {
        // Per-output i8 quantization gives ~0.4 % relative RMS at
        // each layer; two layers compose to ≤ 1 % per output. Use a
        // generous absolute tolerance because the test weights are
        // small magnitudes.
        let tol = 0.05 + 0.01 * a.abs();
        assert!(
            (a - b).abs() <= tol,
            "out[{i}] f32={a} i8={b} diff={} tol={tol}",
            (a - b).abs()
        );
    }
}

#[test]
fn i8_unknown_dtype_byte_is_rejected() {
    // Hand-craft a v1 model with weight_dtype = 3 (not f32/f16/i8).
    // Parser must reject with UnknownWeightDtype.
    let mut buf = alloc::vec::Vec::new();
    buf.extend_from_slice(b"ZNPK");
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&32u16.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // n_inputs
    buf.extend_from_slice(&1u32.to_le_bytes()); // n_outputs
    buf.extend_from_slice(&1u32.to_le_bytes()); // n_layers
    buf.extend_from_slice(&0u64.to_le_bytes()); // schema_hash
    buf.extend_from_slice(&0u32.to_le_bytes()); // flags
    buf.extend_from_slice(&0.0f32.to_le_bytes()); // mean[0]
    buf.extend_from_slice(&1.0f32.to_le_bytes()); // scale[0]
    buf.extend_from_slice(&1u32.to_le_bytes()); // in_dim
    buf.extend_from_slice(&1u32.to_le_bytes()); // out_dim
    buf.push(0); // activation
    buf.push(3); // weight_dtype = 3 (unknown)
    buf.extend_from_slice(&[0, 0]);
    buf.extend_from_slice(&1.0f32.to_le_bytes()); // 1 weight as f32 (won't be reached)
    buf.extend_from_slice(&0.0f32.to_le_bytes()); // bias
    let aligned = AlignedBuf::from_slice(&buf);
    let result = Model::from_bytes(aligned.as_bytes());
    assert!(matches!(
        result,
        Err(crate::error::PickerError::UnknownWeightDtype { byte: 3 })
    ));
}

#[test]
fn f16_weight_dtype_round_trips() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f16(
        &mut buf,
        1,
        &[(1, 3, 0, &[1.0, 2.0, 4.0], &[0.0, 0.0, 0.0])],
        0xCAFEFACE_BAADF00D,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    assert!(matches!(
        model.layers()[0].weights,
        crate::model::WeightStorage::F16(_)
    ));
    let mut picker = Picker::new(model);
    let out = picker.predict(&[1.0]).unwrap();
    assert!((out[0] - 1.0).abs() < 1e-5);
    assert!((out[1] - 2.0).abs() < 1e-5);
    assert!((out[2] - 4.0).abs() < 1e-5);
}

#[test]
fn hybrid_heads_argmin_in_range() {
    // Simulate a 6-output hybrid model: 3 bytes_log + 3 scalar
    // chroma_scale predictions. Argmin over the bytes-log sub-range
    // [0..3] should pick the smallest of {2, 1, 3} → idx 1, ignoring
    // the scalar predictions in [3..6].
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(
        &mut buf,
        1,
        &[(
            1,
            6,
            0,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[2.0, 1.0, 3.0, 0.85, 0.95, 1.05],
        )],
        0,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut picker = Picker::new(model);

    // Bytes head: indices 0..3, all allowed.
    let mask_all = AllowedMask::new(&[true, true, true]);
    assert_eq!(
        picker
            .argmin_masked_in_range(&[0.0], (0, 3), &mask_all, None)
            .unwrap(),
        Some(1),
        "smallest in bytes head [2,1,3] is idx 1"
    );

    // Forbid idx 1, expect idx 0.
    let mask_no1 = AllowedMask::new(&[true, false, true]);
    assert_eq!(
        picker
            .argmin_masked_in_range(&[0.0], (0, 3), &mask_no1, None)
            .unwrap(),
        Some(0)
    );

    // Read the chroma_scale predictions directly (just verifying
    // they're reachable via predict()).
    let out = picker.predict(&[0.0]).unwrap();
    assert_eq!(out.len(), 6);
    assert!((out[3] - 0.85).abs() < 1e-6);
    assert!((out[4] - 0.95).abs() < 1e-6);
    assert!((out[5] - 1.05).abs() < 1e-6);
}

#[test]
fn schema_hash_round_trips() {
    let mut buf = alloc::vec::Vec::new();
    write_v1_model_f32(
        &mut buf,
        1,
        &[(1, 1, 0, &[1.0], &[0.0])],
        0xCAFEBABE_DEADBEEF,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    assert_eq!(model.schema_hash(), 0xCAFEBABE_DEADBEEF);
}
