//! Full producer→consumer loop on REAL baked bytes: a model baked with qualified
//! `name@hex8` columns → loaded → a `zenanalyze_api::Request` built from its columns →
//! negotiated against an `Offer`, **purely by per-feature qualified name**. The unit
//! tests cover each piece; this covers them composed on actual ZNPR bytes.
//!
//! The old `(analyzer_version, feature_defs_version, config_hash)` stamps are still
//! written and round-trip, but they are now **informational** — they do NOT gate reuse
//! (the per-feature code version lives in each qualified name instead).

use zenanalyze_api::{FeatureResult, NamedFeature, Offer, Provenance, Request, Select};
use zenpredict::Model;
use zenpredict_bake::bake_from_json_str;

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// A 3-input model whose `feature_columns` are the given newline-joined string; the three
/// (now informational) stamps are included to prove they survive but don't gate reuse.
fn baked(columns: &str) -> Vec<u8> {
    let json = format!(
        r#"{{
            "schema_hash": 4242,
            "scaler_mean":  [0.0, 0.0, 0.0],
            "scaler_scale": [1.0, 1.0, 1.0],
            "layers": [
                {{"in_dim": 3, "out_dim": 3, "activation": "identity", "dtype": "f32",
                 "weights": [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0],
                 "biases": [0.0, 0.0, 0.0]}}
            ],
            "analyzer_version": "0.2.7",
            "feature_defs_version": 1,
            "feature_config_hash": 0,
            "metadata": [
                {{"key": "zentrain.feature_columns", "type": "utf8", "text": "{columns}"}}
            ]
        }}"#
    );
    bake_from_json_str(&json).unwrap()
}

fn fr(qualified: &'static str, v: f32) -> FeatureResult<'static> {
    FeatureResult::new(NamedFeature::parse(qualified).unwrap(), v)
}

/// Build the picker's want-list straight from the loaded model's qualified columns.
fn parse_wants(model: &Model) -> Vec<NamedFeature<'_>> {
    model
        .feature_columns()
        .filter_map(NamedFeature::parse)
        .collect()
}

#[test]
fn baked_model_builds_request_and_negotiates_offer() {
    let aligned = Aligned(baked(
        "variance@11111111\\nedge_density@22222222\\nuniformity@33333333",
    ));
    let model = Model::from_bytes(&aligned.0).unwrap();
    // the three informational stamps round-tripped
    assert_eq!(model.analyzer_version(), Some("0.2.7"));
    assert_eq!(model.feature_defs_version(), Some(1));

    let wants = parse_wants(&model);
    assert_eq!(wants.len(), 3);
    let req = Request::new(Select::Features(&wants));

    // a matching offer with values in a different order reuses, gathered into column order
    let feats = [
        fr("uniformity@33333333", 30.0),
        fr("variance@11111111", 10.0),
        fr("edge_density@22222222", 20.0),
    ];
    let offer = Offer::new(&feats, Provenance::new("0.2.7"));
    assert!(offer.satisfies(&req));
    assert_eq!(offer.reuse_for(&req).unwrap(), [10.0, 20.0, 30.0]);
}

#[test]
fn reuse_is_per_qualified_name_only() {
    let aligned = Aligned(baked(
        "variance@11111111\\nedge_density@22222222\\nuniformity@33333333",
    ));
    let model = Model::from_bytes(&aligned.0).unwrap();
    let wants = parse_wants(&model);
    let req = Request::new(Select::Features(&wants));

    // a feature at a DIFFERENT code version (its @hash differs) ⇒ miss ⇒ own pass
    let drifted = [
        fr("variance@99999999", 10.0), // drift
        fr("edge_density@22222222", 20.0),
        fr("uniformity@33333333", 30.0),
    ];
    assert!(
        Offer::new(&drifted, Provenance::new("0.2.7"))
            .reuse_for(&req)
            .is_none()
    );

    // an offer whose informational provenance differs (a different analyzer_version / config /
    // descriptor) STILL reuses — provenance never gates reuse; only the qualified names do
    let same = [
        fr("variance@11111111", 1.0),
        fr("edge_density@22222222", 1.0),
        fr("uniformity@33333333", 1.0),
    ];
    let other = Provenance::new("9.9.9")
        .with_config(0xBEEF)
        .with_descriptor(0xF00D);
    assert!(Offer::new(&same, other).satisfies(&req));
}

#[test]
fn stamps_are_informational_not_a_reuse_gate() {
    // A bake without the old stamps: accessors are None, yet reuse still works by qualified
    // name — confirming the stamps never gated it.
    let json = r#"{
        "schema_hash": 1,
        "scaler_mean": [0.0], "scaler_scale": [1.0],
        "layers": [{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32",
                    "weights":[1.0],"biases":[0.0]}],
        "metadata": [{"key":"zentrain.feature_columns","type":"utf8","text":"variance@11111111"}]
    }"#;
    let aligned = Aligned(bake_from_json_str(json).unwrap());
    let model = Model::from_bytes(&aligned.0).unwrap();
    assert_eq!(model.analyzer_version(), None);
    assert_eq!(model.feature_defs_version(), None);
    assert_eq!(model.feature_config_hash(), None);

    let wants = parse_wants(&model);
    let req = Request::new(Select::Features(&wants));
    let feats = [fr("variance@11111111", 1.0)];
    assert!(Offer::new(&feats, Provenance::new("0.2.7")).satisfies(&req));
}

fn tmp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("zpbake-fc-{}-{}", std::process::id(), name));
    p
}

#[test]
fn repack_stamps_an_unstamped_bin() {
    use zenpredict_bake::cli::run_repack_cli;

    // An UNSTAMPED bake (a pre-contract codec picker), with feature columns.
    let json = r#"{
        "schema_hash": 555,
        "scaler_mean":  [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [{"in_dim": 2, "out_dim": 2, "activation": "identity", "dtype": "f32",
                    "weights": [1.0,0.0, 0.0,1.0], "biases": [0.0, 0.0]}],
        "metadata": [
            {"key": "zentrain.feature_columns", "type": "utf8", "text": "variance\nedge_density"}
        ]
    }"#;
    let unstamped = bake_from_json_str(json).unwrap();
    let in_path = tmp_path("repack-in.bin");
    let out_path = tmp_path("repack-out.bin");
    std::fs::write(&in_path, &unstamped).unwrap();

    // The codec re-bake path: stamp the existing bin (no re-training).
    let argv: Vec<String> = [
        in_path.to_string_lossy().as_ref(),
        out_path.to_string_lossy().as_ref(),
        "--analyzer-version",
        "0.2.7",
        "--feature-defs-version",
        "1",
        "--config-hash",
        "0",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let _ = run_repack_cli(&argv); // ExitCode isn't PartialEq; verify via the output bin.

    let out_bytes = std::fs::read(&out_path).expect("repack must write the output bin");
    let aligned = Aligned(out_bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();

    // Stamps written (informational)...
    assert_eq!(model.analyzer_version(), Some("0.2.7"));
    assert_eq!(model.feature_defs_version(), Some(1));
    assert_eq!(model.feature_config_hash(), Some(0));
    // ...and the original content survived the round-trip.
    let cols: Vec<&str> = model.feature_columns().collect();
    assert_eq!(cols, ["variance", "edge_density"]);
    assert_eq!(model.schema_hash(), 555);

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
}
