//! Live check of the shipped cross-codec routers: real images → a zenanalyze offer from
//! this build → `zenpicker::default_route`. The decision must be a model route or the
//! explicit, reported heuristic fallback; a silent `None` (the pre-2026-10-10 behaviour
//! when a router's pinned features drifted) fails the test.
//!
//! Images come from codec-corpus `gb82` (25 CC0 photographs, 576×576), downloaded on first
//! use; a failed download fails the test. Run with `--features api`.
#![cfg(feature = "api")]

use std::collections::BTreeSet;

use zenanalyze_api::{Offer, OwnedFeatureResult, Request, Select};
use zenpicker::{
    AllowedFamilies, CodecFamily, FallbackReason, QualityTarget, RouteSource, family_rule,
};
use zenpredict::EncodeMode;

/// Router columns this build cannot supply at the pinned version, as of 2026-10-10:
/// `chroma_subsample_dct_loss` drifted on 2026-07-03/05 and the chroma–luma covariances
/// changed in feature-defs version 2 (ZANFIX). While this set is non-empty every live image
/// takes the heuristic path. A retrain on current features empties it; a new drift grows
/// it. Either way this test fails: update this set and the "Router status" section of
/// `zenpicker/README.md` together.
const KNOWN_DRIFT: [&str; 3] = [
    "chroma_luma_covariance_cb",
    "chroma_luma_covariance_cr",
    "chroma_subsample_dct_loss",
];

fn gb82_images() -> Vec<(String, image::RgbImage)> {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus cache");
    let dir = corpus.get("gb82").expect("download codec-corpus gb82");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("read gb82")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")))
        .collect();
    paths.sort();
    assert!(
        paths.len() >= 20,
        "gb82 should hold 25 PNGs, found {} in {}",
        paths.len(),
        dir.display()
    );
    paths
        .iter()
        .map(|p| {
            let img = image::open(p)
                .unwrap_or_else(|e| panic!("decode {}: {e}", p.display()))
                .to_rgb8();
            (p.file_name().unwrap().to_string_lossy().into_owned(), img)
        })
        .collect()
}

#[test]
fn default_route_on_live_offers_is_a_model_route_or_a_reported_fallback() {
    let est = [0u32; CodecFamily::COUNT];
    let mode = EncodeMode::QueuedBalanced;
    let targets = [
        QualityTarget::Zq(40.0),
        QualityTarget::Zq(80.0),
        QualityTarget::Lossless,
    ];
    let (mut model, mut heuristic) = (0usize, 0usize);
    let mut drifted: BTreeSet<String> = BTreeSet::new();
    let mut first_reason: Option<String> = None;
    for (name, img) in gb82_images() {
        let (w, h) = img.dimensions();
        let owned = zenanalyze::offer_for_request(img.as_raw(), w, h, &Request::new(Select::All))
            .unwrap_or_else(|e| panic!("{name}: analysis failed: {e}"));
        let cells: Vec<_> = owned
            .features()
            .iter()
            .map(OwnedFeatureResult::as_ref)
            .collect();
        let offer = Offer::new(&cells, owned.provenance());
        for target in targets {
            let decision =
                zenpicker::default_route(&offer, target, &CodecFamily::ALL, mode, None, &est)
                    .unwrap_or_else(|e| panic!("{name} {target:?}: route error {e:?}"))
                    .unwrap_or_else(|| {
                        panic!("{name} {target:?}: silent None with every family allowed")
                    });
            assert_eq!(decision.ranked().first(), Some(&decision.family()));
            match decision.source() {
                RouteSource::Model => model += 1,
                RouteSource::Heuristic(reason) => {
                    heuristic += 1;
                    let rule =
                        family_rule(&offer, target, AllowedFamilies::all(), mode, None, &est);
                    assert_eq!(
                        Some(decision.family()),
                        rule,
                        "{name} {target:?}: the fallback must be family_rule"
                    );
                    let FallbackReason::FeatureDrift { features, .. } = reason else {
                        panic!("{name} {target:?}: unexpected fallback reason {reason}");
                    };
                    assert!(!features.is_empty(), "{name}: drift with no features");
                    for m in features {
                        let bare = m.wanted.split('@').next().unwrap();
                        let have = offer.get(bare).map(|f| f.feature().qualified_name());
                        assert_eq!(
                            have,
                            m.offered.as_deref(),
                            "{name}: reported offer identity must be the offer's"
                        );
                        assert_ne!(have, Some(m.wanted.as_str()), "{name}: not a drift");
                        drifted.insert(bare.to_string());
                    }
                    first_reason.get_or_insert_with(|| reason.to_string());
                }
                other => panic!("{name} {target:?}: unknown route source {other:?}"),
            }
        }
    }
    eprintln!("zenpicker live routes: {model} model, {heuristic} heuristic; drifted: {drifted:?}");
    if let Some(reason) = &first_reason {
        eprintln!("first fallback reason: {reason}");
    }
    let known: BTreeSet<String> = KNOWN_DRIFT.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        drifted, known,
        "router feature drift changed: update KNOWN_DRIFT and zenpicker/README.md \"Router status\""
    );
    if known.is_empty() {
        assert_eq!(heuristic, 0, "no drift, yet the heuristic path ran");
    } else {
        assert_eq!(model, 0, "known drift, yet a model route ran");
    }
}
