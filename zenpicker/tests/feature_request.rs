//! `MetaPicker::feature_request` builds a `zenanalyze_api::Request` from the
//! picker's baked model that a caller negotiates against a shared `Offer` before
//! `pick`. Requires the `api` feature.
#![cfg(feature = "api")]

use zenanalyze_api::Offer;
use zenpicker::MetaPicker;
use zenpredict::Model;
use zenpredict_bake::bake_from_json_str;

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// A 2-feature model stamped with the full reuse key.
fn model_bytes() -> Vec<u8> {
    let json = r#"{
        "schema_hash": 77,
        "scaler_mean":  [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [{"in_dim": 2, "out_dim": 2, "activation": "identity", "dtype": "f32",
                    "weights": [1.0,0.0, 0.0,1.0], "biases": [0.0, 0.0]}],
        "analyzer_version": "0.2.7",
        "feature_defs_version": 1,
        "feature_config_hash": 0,
        "metadata": [
            {"key": "zentrain.feature_columns", "type": "utf8", "text": "variance\nedge_density"}
        ]
    }"#;
    bake_from_json_str(json).unwrap()
}

#[test]
fn feature_request_negotiates_a_shared_offer() {
    let aligned = Aligned(model_bytes());
    let model = Model::from_bytes(&aligned.0).unwrap();
    let picker = MetaPicker::new(&model);

    let req = picker.feature_request();

    // A matching shared offer (names in a different order) reuses, gathered into
    // the model's column order [variance, edge_density].
    let offer = Offer::new(&["edge_density", "variance"], &[2.0, 1.0], "0.2.7", 1, 0);
    let v = offer
        .reuse_for(&req)
        .expect("matching stamps + coverage must reuse");
    assert_eq!(v, [1.0, 2.0]);

    // A linear-light offer (config drift) against this gamma-trained model:
    // own-pass, never a wrong reuse.
    let linear = Offer::new(&["variance", "edge_density"], &[9.0, 9.0], "0.2.7", 1, 0x99);
    assert!(linear.reuse_for(&req).is_none());

    // A newer-minor offer: own-pass.
    let newer = Offer::new(&["variance", "edge_density"], &[9.0, 9.0], "0.3.0", 1, 0);
    assert!(newer.reuse_for(&req).is_none());
}
