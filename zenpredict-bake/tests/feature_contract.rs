//! Full producer→consumer loop on REAL baked bytes.
//!
//! Proves the whole `zenanalyze-api` chain holds across the bake boundary: a
//! model baked with the three reuse-key stamps → loaded → a
//! `zenanalyze_api::Request` built *from the model's own accessors* (the real
//! consumer path, not a hand-built request) → negotiated against an `Offer`
//! exactly as the reuse key dictates. The unit tests cover each piece; this
//! covers them composed on actual ZNPR bytes.

use zenanalyze_api::{Offer, Request};
use zenpredict::Model;
use zenpredict_bake::bake_from_json_str;

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// A 3-input model stamped with feature columns + the full reuse key.
fn baked_model_json() -> &'static str {
    r#"{
        "schema_hash": 4242,
        "scaler_mean":  [0.0, 0.0, 0.0],
        "scaler_scale": [1.0, 1.0, 1.0],
        "layers": [
            {"in_dim": 3, "out_dim": 3, "activation": "identity", "dtype": "f32",
             "weights": [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0],
             "biases": [0.0, 0.0, 0.0]}
        ],
        "analyzer_version": "0.2.7",
        "feature_defs_version": 1,
        "feature_config_hash": 0,
        "metadata": [
            {"key": "zentrain.feature_columns", "type": "utf8",
             "text": "variance\nedge_density\nuniformity"}
        ]
    }"#
}

/// Build a `Request` the way a real codec does: straight from the loaded model's
/// self-describing metadata.
fn request_from_model<'a>(model: &'a Model, names: &'a [&'a str]) -> Request<'a> {
    Request::new(
        names,
        model.analyzer_version().unwrap_or(""),
        model.feature_defs_version().unwrap_or(0),
        model.feature_config_hash().unwrap_or(0),
    )
}

#[test]
fn baked_model_builds_request_and_negotiates_offer() {
    let bytes = bake_from_json_str(baked_model_json()).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();

    // The consumer reads its feature names + reuse key off the model itself.
    let names: Vec<&str> = model.feature_columns().collect();
    assert_eq!(names, ["variance", "edge_density", "uniformity"]);
    let req = request_from_model(&model, &names);

    // An offer a matching zenanalyze pass would produce (same names + reuse key,
    // values in a different order to prove gather reorders by name).
    let offer_names = ["uniformity", "variance", "edge_density"];
    let offer_vals = [30.0, 10.0, 20.0];
    let offer = Offer::new(&offer_names, &offer_vals, "0.2.7", 1, 0);

    let v = req_reuse(&offer, &req).expect("matching baked stamps + coverage must reuse");
    // Gathered in the model's column order: variance, edge_density, uniformity.
    assert_eq!(v, [10.0, 20.0, 30.0]);
}

#[test]
fn baked_model_rejects_mismatched_offers() {
    let bytes = bake_from_json_str(baked_model_json()).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let names: Vec<&str> = model.feature_columns().collect();
    let req = request_from_model(&model, &names);

    let offer_names = ["variance", "edge_density", "uniformity"];
    let offer_vals = [10.0, 20.0, 30.0];

    // defs_version drift → own pass.
    let defs_drift = Offer::new(&offer_names, &offer_vals, "0.2.7", 2, 0);
    assert!(
        req_reuse(&defs_drift, &req).is_none(),
        "defs drift must not reuse"
    );

    // analysis-config drift (e.g. linear-light offer vs gamma-trained model) → own pass.
    let cfg_drift = Offer::new(&offer_names, &offer_vals, "0.2.7", 1, 0xBEEF);
    assert!(
        req_reuse(&cfg_drift, &req).is_none(),
        "config drift must not reuse"
    );

    // major version drift → own pass.
    let major_drift = Offer::new(&offer_names, &offer_vals, "1.0.0", 1, 0);
    assert!(
        req_reuse(&major_drift, &req).is_none(),
        "major drift must not reuse"
    );

    // patch-only difference → reuse (patch isn't part of the key).
    let patch_ok = Offer::new(&offer_names, &offer_vals, "0.2.99", 1, 0);
    assert!(
        req_reuse(&patch_ok, &req).is_some(),
        "patch difference must still reuse"
    );
}

#[test]
fn model_without_stamps_defaults_to_zero_key() {
    // A bake predating the stamps: accessors are None, the consumer falls back to
    // (analyzer_version="", defs=0, config=0). Such a request only matches an
    // equally-unstamped offer — so it safely never reuses a real versioned offer.
    let json = r#"{
        "schema_hash": 1,
        "scaler_mean": [0.0], "scaler_scale": [1.0],
        "layers": [{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32",
                    "weights":[1.0],"biases":[0.0]}],
        "metadata": [{"key":"zentrain.feature_columns","type":"utf8","text":"variance"}]
    }"#;
    let bytes = bake_from_json_str(json).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    assert_eq!(model.analyzer_version(), None);
    assert_eq!(model.feature_defs_version(), None);
    assert_eq!(model.feature_config_hash(), None);

    let names: Vec<&str> = model.feature_columns().collect();
    let req = request_from_model(&model, &names);

    // A real versioned offer does NOT satisfy the unstamped request → own pass.
    let real = Offer::new(&["variance"], &[1.0], "0.2.7", 1, 0);
    assert!(req_reuse(&real, &req).is_none());
}

/// Helper mirroring `Offer::reuse_for` but spelled out so the lifetimes are
/// obvious in the assertions above.
fn req_reuse(offer: &Offer<'_>, req: &Request<'_>) -> Option<Vec<f32>> {
    offer.reuse_for(req)
}
