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

/// Build a v1 model with f32 weights into `out`.
///
/// `mean[i] = 0.0`, `scale[i] = 1.0` (identity scaler) for tests.
fn write_v1_model_f32(
    out: &mut alloc::vec::Vec<u8>,
    n_inputs: usize,
    layers: &[(usize, usize, u8, &[f32], &[f32])], // (in, out, activation, W row-major, b)
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
    write_v1_model_f32(
        &mut buf,
        1,
        &[(1, 2, 1, &[-2.0, 1.0], &[0.0, 0.0])],
        0,
    );
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
                &[
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                    1.0, 1.0, 1.0,
                ],
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
    assert_eq!(picker.argmin_masked(&[0.0], &mask_none, None).unwrap(), None);
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
