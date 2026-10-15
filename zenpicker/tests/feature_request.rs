//! `MetaPicker::feature_request` builds a `Select::Features` of qualified
//! `NamedFeature`s from the picker's baked model; a caller negotiates it against a
//! shared `Offer` before `pick`. Requires the `api` feature.
#![cfg(feature = "api")]

use zenanalyze_api::{FeatureResult, NamedFeature, Offer, Provenance};
use zenpicker::{
    AllowedFamilies, CodecFamily, FallbackReason, MetaPicker, QualityTarget, RouteSource,
    RouterKind, family_rule,
};
use zenpredict::Model;
use zenpredict_bake::bake_from_json_str;

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// A 2-feature model whose `feature_columns` are the given newline-joined string.
fn baked(columns: &str) -> Vec<u8> {
    let json = format!(
        r#"{{
        "schema_hash": 77,
        "scaler_mean":  [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [{{"in_dim": 2, "out_dim": 2, "activation": "identity", "dtype": "f32",
                    "weights": [1.0,0.0, 0.0,1.0], "biases": [0.0, 0.0]}}],
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

#[test]
fn feature_request_negotiates_a_shared_offer() {
    // columns are QUALIFIED `name@hex8` — each carries its per-feature code version
    let aligned = Aligned(baked("variance@11111111\\nedge_density@22222222"));
    let model = Model::from_bytes(&aligned.0).unwrap();
    let picker = MetaPicker::new(&model);
    let req = picker
        .feature_request()
        .expect("fully-qualified columns ⇒ reusable");

    // a matching offer (qualified names reordered) reuses, gathered into column order
    let feats = [
        fr("edge_density@22222222", 2.0),
        fr("variance@11111111", 1.0),
    ];
    let offer = Offer::new(&feats, Provenance::new("0.2.7"));
    assert!(offer.satisfies(&req));
    assert_eq!(
        offer.reuse_for(&req).expect("matching columns reuse"),
        [1.0, 2.0]
    );

    // an offer whose `variance` is a different CODE version (drift) ⇒ miss ⇒ own pass
    let drifted = [
        fr("variance@99999999", 9.0),
        fr("edge_density@22222222", 9.0),
    ];
    let drifted = Offer::new(&drifted, Provenance::new("0.2.7"));
    assert!(!drifted.satisfies(&req));
    assert!(drifted.reuse_for(&req).is_none());

    // a missing column ⇒ own pass, never a silent zero
    let partial = [fr("variance@11111111", 1.0)];
    assert!(
        Offer::new(&partial, Provenance::new("0.2.7"))
            .reuse_for(&req)
            .is_none()
    );
}

#[test]
fn legacy_bare_name_bake_cannot_reuse() {
    // pre-`name@hash` columns don't parse ⇒ feature_request is None ⇒ always own-pass
    let aligned = Aligned(baked("variance\\nedge_density"));
    let model = Model::from_bytes(&aligned.0).unwrap();
    let picker = MetaPicker::new(&model);
    assert!(picker.feature_request().is_none());
}

#[test]
fn unqualified_routers_fall_back_with_the_reason() {
    // Pre-`name@hash` columns: no router can reuse an offer, so route() misses and
    // route_or_heuristic reports UnqualifiedColumns for the router that ran first.
    let bytes = Aligned(baked("variance\\nedge_density"));
    let lossy = Model::from_bytes(&bytes.0).unwrap();
    let gate = Model::from_bytes(&bytes.0).unwrap();
    let lossless = Model::from_bytes(&bytes.0).unwrap();
    let mut picker = MetaPicker::new(&lossy).with_router(&gate, &lossless);
    let feats = [
        fr("variance@11111111", 1.0),
        fr("edge_density@22222222", 2.0),
    ];
    let offer = Offer::new(&feats, Provenance::new("0.2.7"));
    let est = [0u32; CodecFamily::COUNT];
    let mode = zenpredict::EncodeMode::QueuedBalanced;
    for (target, router) in [
        (QualityTarget::Zq(80.0), RouterKind::Gate),
        (QualityTarget::Lossless, RouterKind::Lossless),
    ] {
        let all = AllowedFamilies::all();
        assert_eq!(
            picker.route(&offer, target, all, mode, None, &est).unwrap(),
            None
        );
        let d = picker
            .route_or_heuristic(&offer, target, all, mode, None, &est)
            .unwrap()
            .expect("heuristic decision");
        assert_eq!(
            Some(d.family()),
            family_rule(&offer, target, all, mode, None, &est)
        );
        let RouteSource::Heuristic(FallbackReason::UnqualifiedColumns { router: r, .. }) =
            d.source()
        else {
            panic!("expected UnqualifiedColumns, got {:?}", d.source());
        };
        assert_eq!(*r, router);
        let RouteSource::Heuristic(reason) = d.source() else {
            unreachable!()
        };
        assert_eq!(
            reason.to_string(),
            format!(
                "{} router disabled: its columns are not qualified name@hash identities",
                router.label()
            )
        );
    }
}
