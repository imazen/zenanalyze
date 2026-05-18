//! Integration tests for the 5 stacked `FeatureTransform` variants
//! added 2026-05-17 (WinsorThenLog / WinsorThenLog1p /
//! WinsorThenSignedCbrt / SignedCbrtThenWinsor / ClipThenLog1pThenWinsor).
//!
//! Coverage:
//! - Token round-trip via `Model::from_bytes` for each variant.
//! - End-to-end bake + predict applies the transform value-correctly
//!   given valid per-feature params (4-input passthrough model).
//! - Bake-side validator rejects per-variant arity mismatches.
//! - Bake-side validator rejects per-variant domain violations
//!   (`WinsorThenLog` `p1<=0`, `WinsorThenLog1p` `p1<=-1`, etc.).
//! - Apply-without-params fallback at runtime degrades gracefully
//!   when the bake omits the params blob (legacy bakes).

use zenpredict::{Activation, FeatureTransform, MetadataType, Model, Predictor, WeightDtype, keys};
use zenpredict_bake::{BakeError, BakeLayer, BakeMetadataEntry, BakeRequest, bake};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// 4-input → 4-output identity-passthrough model. Each output `i`
/// forwards `features[i]` unchanged, so `predict_transformed(raw)[i]`
/// equals `transforms[i].apply_with_params(raw[i], params[i])`.
fn make_identity_passthrough(metadata: &[BakeMetadataEntry<'_>]) -> Result<Vec<u8>, BakeError> {
    let scaler_mean = [0.0f32; 4];
    let scaler_scale = [1.0f32; 4];
    let w0 = [
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let b0 = [0.0f32; 4];
    let layers = [BakeLayer {
        in_dim: 4,
        out_dim: 4,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &w0,
        biases: &b0,
    }];
    bake(&BakeRequest {
        schema_hash: 0xfeed_face_dead_beef,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
}

#[test]
fn all_five_stacks_round_trip_through_bake() {
    // Sanity: each new token can be baked into a `feature_transforms`
    // blob and read back via `model.feature_transforms()` as the
    // matching variant. Each row's params are arity-correct so the
    // bake-side validator accepts.
    let txt =
        b"winsor_then_log\nwinsor_then_log1p\nwinsor_then_signed_cbrt\nclip_then_log1p_then_winsor";
    // p1>0 for winsor_then_log; p1>-1 for winsor_then_log1p;
    // [eps>=0, q1>=0, q99] for clip_then_log1p_then_winsor.
    let params_txt = b"0.5,99.0\n0.0,50.0\n-100.0,100.0\n0.5,0.0,5.0";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let bytes = make_identity_passthrough(&metadata).expect("bake should accept valid stacks");
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let parsed = model.feature_transforms().expect("transforms present");
    assert_eq!(
        parsed,
        &[
            FeatureTransform::WinsorThenLog,
            FeatureTransform::WinsorThenLog1p,
            FeatureTransform::WinsorThenSignedCbrt,
            FeatureTransform::ClipThenLog1pThenWinsor,
        ]
    );
}

#[test]
fn signed_cbrt_then_winsor_round_trip() {
    // SignedCbrtThenWinsor uses cbrt-domain bounds. Pair it with
    // three identities to keep the 4-row contract.
    let txt = b"identity\nsigned_cbrt_then_winsor\nidentity\nidentity";
    let params_txt = b"\n-1.5,1.5\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let bytes = make_identity_passthrough(&metadata).expect("bake should accept");
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let parsed = model.feature_transforms().expect("present");
    assert_eq!(parsed[1], FeatureTransform::SignedCbrtThenWinsor);
}

#[test]
fn winsor_then_log_predicts_correct_value() {
    // 1-row test: feature[0] uses winsor_then_log with [1.0, 100.0].
    // Feeding x=1000 should clip to 100, then ln(100) ≈ 4.6052.
    let txt = b"winsor_then_log\nidentity\nidentity\nidentity";
    let params_txt = b"1.0,100.0\n\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let bytes = make_identity_passthrough(&metadata).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);
    let raw = [1000.0f32, 0.0, 0.0, 0.0];
    let out = predictor.predict_transformed(&raw).unwrap().to_vec();
    // ln(100) ≈ 4.605170
    let expected = 100f32.ln();
    assert!(
        (out[0] - expected).abs() < 1e-4,
        "got {} expected {expected}",
        out[0]
    );

    // Below p1=1.0 should clip up to 1.0 → ln(1) = 0.
    let raw_low = [0.5f32, 0.0, 0.0, 0.0];
    let out_low = predictor.predict_transformed(&raw_low).unwrap().to_vec();
    assert!(out_low[0].abs() < 1e-5, "ln(1)=0, got {}", out_low[0]);
}

#[test]
fn clip_then_log1p_then_winsor_predicts_correct_value() {
    // Three-step pipeline: eps=2, q1=0, q99=3.
    let txt = b"clip_then_log1p_then_winsor\nidentity\nidentity\nidentity";
    let params_txt = b"2.0,0.0,3.0\n\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let bytes = make_identity_passthrough(&metadata).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);
    // x=10: shifted=8, ln1p(8) ≈ 2.197 — within [0,3], unchanged.
    let raw = [10.0f32, 0.0, 0.0, 0.0];
    let out = predictor.predict_transformed(&raw).unwrap().to_vec();
    let expected = 8f32.ln_1p();
    assert!(
        (out[0] - expected).abs() < 1e-4,
        "got {} expected {expected}",
        out[0]
    );
    // x=10000: shifted=9998, ln1p(9998) ≈ 9.21 — clipped to 3.0.
    let raw_hi = [10000.0f32, 0.0, 0.0, 0.0];
    let out_hi = predictor.predict_transformed(&raw_hi).unwrap().to_vec();
    assert!(
        (out_hi[0] - 3.0).abs() < 1e-5,
        "clip to q99=3.0, got {}",
        out_hi[0]
    );
}

#[test]
fn winsor_then_log_rejects_p1_zero_at_bake() {
    // p1=0 produces ln(0) = -Inf at runtime — bake validator rejects.
    let txt = b"winsor_then_log\nidentity\nidentity\nidentity";
    let params_txt = b"0.0,100.0\n\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let err = make_identity_passthrough(&metadata).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::FeatureTransformParamInvalid {
                feature_index: 0,
                transform: "winsor_then_log",
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn winsor_then_log1p_rejects_p1_too_low_at_bake() {
    // p1=-1.5 produces ln1p(<0) = NaN at runtime — bake validator
    // rejects.
    let txt = b"winsor_then_log1p\nidentity\nidentity\nidentity";
    let params_txt = b"-1.5,100.0\n\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let err = make_identity_passthrough(&metadata).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::FeatureTransformParamInvalid {
                feature_index: 0,
                transform: "winsor_then_log1p",
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn winsor_then_signed_cbrt_rejects_inverted_bounds() {
    // p1 > p99 is rejected (math would produce a single-valued clamp,
    // almost certainly a transcription bug).
    let txt = b"winsor_then_signed_cbrt\nidentity\nidentity\nidentity";
    let params_txt = b"100.0,1.0\n\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let err = make_identity_passthrough(&metadata).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::FeatureTransformParamInvalid {
                feature_index: 0,
                transform: "winsor_then_signed_cbrt",
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn clip_then_log1p_then_winsor_rejects_negative_eps() {
    // eps must be >= 0; the noise-floor subtract goes weird with
    // negative input.
    let txt = b"clip_then_log1p_then_winsor\nidentity\nidentity\nidentity";
    let params_txt = b"-1.0,0.0,3.0\n\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let err = make_identity_passthrough(&metadata).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::FeatureTransformParamInvalid {
                feature_index: 0,
                transform: "clip_then_log1p_then_winsor",
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn winsor_then_log_rejects_wrong_arity() {
    // Only 1 param provided; needs 2.
    let txt = b"winsor_then_log\nidentity\nidentity\nidentity";
    let params_txt = b"1.0\n\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let err = make_identity_passthrough(&metadata).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::FeatureTransformParamArityMismatch {
                feature_index: 0,
                transform: "winsor_then_log",
                expected: 2,
                got: 1,
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn clip_then_log1p_then_winsor_rejects_wrong_arity() {
    // Only 2 params provided; needs 3.
    let txt = b"clip_then_log1p_then_winsor\nidentity\nidentity\nidentity";
    let params_txt = b"0.5,1.0\n\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let err = make_identity_passthrough(&metadata).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::FeatureTransformParamArityMismatch {
                feature_index: 0,
                transform: "clip_then_log1p_then_winsor",
                expected: 3,
                got: 2,
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn signed_cbrt_then_winsor_applies_in_cbrt_space() {
    // q1, q99 are bounds in cbrt-space — signed_cbrt(8)=2 should
    // clip down to 1.5 when q99=1.5.
    let txt = b"signed_cbrt_then_winsor\nidentity\nidentity\nidentity";
    let params_txt = b"-1.5,1.5\n\n\n";
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params_txt,
        },
    ];
    let bytes = make_identity_passthrough(&metadata).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);
    let raw = [8.0f32, 0.0, 0.0, 0.0];
    let out = predictor.predict_transformed(&raw).unwrap().to_vec();
    assert!((out[0] - 1.5).abs() < 1e-5, "expected 1.5, got {}", out[0]);
}

#[test]
fn stacks_without_params_blob_fall_back_gracefully() {
    // Bake with `feature_transforms` but no `feature_transform_params`
    // — runtime parses transforms but params are absent. The
    // `apply()` fallback path (no params) kicks in and produces the
    // documented degenerate behaviour without panicking. The
    // bake-side validator allows this (params are optional).
    let txt = b"winsor_then_log\nwinsor_then_log1p\nidentity\nidentity";
    let metadata = [BakeMetadataEntry {
        key: keys::FEATURE_TRANSFORMS,
        kind: MetadataType::Utf8,
        value: txt,
    }];
    let bytes = make_identity_passthrough(&metadata).expect("bake should accept");
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);
    let raw = [10.0f32, 5.0, 1.0, 2.0];
    // Should not panic; just check the call succeeds.
    let out = predictor.predict_transformed(&raw).unwrap().to_vec();
    // Without params, WinsorThenLog falls back to plain Log; here
    // ln(10) ≈ 2.302.
    let expected = 10f32.ln();
    assert!(
        (out[0] - expected).abs() < 1e-4,
        "fallback should be plain log; got {}",
        out[0]
    );
    let expected1 = 5f32.ln_1p();
    assert!(
        (out[1] - expected1).abs() < 1e-4,
        "fallback should be plain log1p; got {}",
        out[1]
    );
}
