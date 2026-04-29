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
