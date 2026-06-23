//! Producing a [`zenanalyze_api::OwnedOffer`] from an analysis pass — the producer
//! side of the version-unifying feature contract (gated by the `api` feature).
//!
//! [`extract_offer`] runs **one** analysis pass and bundles the result as a
//! [`zenanalyze_api::OwnedOffer`]: every computed feature as a `name@hex8` qualified
//! cell (its **code** version folded into the name) plus the per-offer
//! [`Provenance`](zenanalyze_api::Provenance). The orchestrator runs one pass per
//! `(config, framing)` group and hands the offer to every codec in the group; each
//! codec negotiates with [`OwnedOffer::satisfies`](zenanalyze_api::OwnedOffer::satisfies)
//! / [`reuse_for`](zenanalyze_api::OwnedOffer::reuse_for) — or, to lend one borrowed
//! [`Offer`](zenanalyze_api::Offer) to many codecs zero-cost, materialize it once:
//! `Offer::new(&owned.features().iter().map(OwnedFeatureResult::as_ref).collect::<Vec<_>>(), owned.provenance())`.
//!
//! The owned storage lives in the frozen `zenanalyze-api`; this module just maps an
//! analysis result into it, folding `zenanalyze::feature_version_hash` into each
//! qualified name via [`NamedFeature::fold_hash`](zenanalyze_api::NamedFeature::fold_hash).

use crate::feature::{AnalysisQuery, FeatureValue};
use crate::versioning::feature_version_hash;
use crate::{analyze_features_rgb8, analyzer_version, feature_name};
use zenanalyze_api::{NamedFeature, OwnedFeatureResult, OwnedOffer, Provenance, Value};

/// Project zenanalyze's native [`FeatureValue`] onto the contract's [`Value`], preserving the
/// exact type. Exhaustive in-crate — adding a `FeatureValue` variant forces a decision here on
/// how the new native type maps onto the contract.
fn to_api_value(v: FeatureValue) -> Value {
    match v {
        FeatureValue::F32(x) => Value::F32(x),
        FeatureValue::U32(x) => Value::U32(x),
        FeatureValue::U64(x) => Value::U64(x),
        FeatureValue::Bool(b) => Value::Bool(b),
    }
}

/// Run **one** analysis pass for `query` over an RGB8 buffer and bundle it as an offerable
/// [`zenanalyze_api::OwnedOffer`].
///
/// The offer carries exactly the features in `query.features()` that the pass computed AND
/// that have a golden version row (unversioned features are omitted, so a consumer that
/// needs them gets a miss and runs its own pass rather than a silent zero). Each feature's
/// name is qualified with the folded ([`NamedFeature::fold_hash`]) `feature_version_hash` —
/// its build-stable **code** version — so two analyzer versions whose math agrees produce
/// the same qualified name and reuse across versions.
///
/// `descriptor_hash` is the value-affecting input framing
/// ([`crate::versioning::descriptor_hash_of`], or
/// [`crate::versioning::rgb8_srgb_descriptor_hash`] for the RGB8 fast path); it rides on the
/// [`Provenance`] for the serialization blend gate, not the per-feature reuse key. `rgb.len()`
/// must be `width * height * 3` (same contract as [`analyze_features_rgb8`]).
#[must_use]
pub fn extract_offer(
    rgb: &[u8],
    width: u32,
    height: u32,
    query: &AnalysisQuery,
    descriptor_hash: u64,
) -> OwnedOffer {
    let results = analyze_features_rgb8(rgb, width, height, query);
    let mut cells = Vec::new();
    for feat in query.features().iter() {
        if let (Some(name), Some(version), Some(value)) = (
            feature_name(feat.id()),
            feature_version_hash(feat),
            results.get(feat),
        ) {
            let qualified = NamedFeature::qualified_for(name, NamedFeature::fold_hash(version));
            cells.push(OwnedFeatureResult::new(&qualified, to_api_value(value)));
        }
    }
    let provenance = Provenance::new(analyzer_version())
        .with_config(query.config_hash())
        .with_descriptor(descriptor_hash);
    OwnedOffer::new(cells, provenance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{AnalysisFeature, FeatureSet};
    use crate::versioning::rgb8_srgb_descriptor_hash;
    use zenanalyze_api::{NamedFeature, Offer, OwnedFeatureResult, Request, Select, Value};

    fn img(w: u32, h: u32) -> Vec<u8> {
        (0..w * h * 3).map(|i| (i % 251) as u8).collect()
    }

    /// The whole contract end to end: zenanalyze runs a pass, lends an offer, and a
    /// consumer's `Request` (built from the producer's own qualified names) reuses it —
    /// exactly as the per-feature reuse key dictates.
    #[test]
    fn extract_offer_round_trips_through_reuse() {
        let (w, h) = (64u32, 64u32);
        let query = AnalysisQuery::new(
            FeatureSet::just(AnalysisFeature::Variance).with(AnalysisFeature::EdgeDensity),
        );
        let owned = extract_offer(&img(w, h), w, h, &query, rgb8_srgb_descriptor_hash());

        // The offer carries qualified names — `variance@<folded code version>`, etc.
        let variance_q = {
            let v = feature_version_hash(AnalysisFeature::Variance).unwrap();
            NamedFeature::qualified_for("variance", NamedFeature::fold_hash(v))
        };
        let edge_q = {
            let v = feature_version_hash(AnalysisFeature::EdgeDensity).unwrap();
            NamedFeature::qualified_for("edge_density", NamedFeature::fold_hash(v))
        };
        assert!(owned.get("variance").is_some());
        assert_eq!(owned.get("variance").unwrap().qualified_name(), variance_q);

        // A consumer requesting the SAME qualified names (subset, reordered) reuses, in order.
        let wants = [
            NamedFeature::parse(&edge_q).unwrap(),
            NamedFeature::parse(&variance_q).unwrap(),
        ];
        let req = Request::new(Select::Features(&wants));
        assert!(owned.satisfies(&req));
        let v = owned
            .reuse_for(&req)
            .expect("same version + coverage must reuse");
        assert_eq!(v.len(), 2);

        // A want at a DIFFERENT code version (drift) ⇒ miss ⇒ own pass.
        let drift = [NamedFeature::parse("variance@ffffffff").unwrap()];
        assert!(!owned.satisfies(&Request::new(Select::Features(&drift))));
        // A name the offer doesn't carry ⇒ miss, never a silent zero.
        let missing = [NamedFeature::parse(&variance_q).unwrap(), {
            let q = "noise_floor_y@00000000";
            NamedFeature::parse(q).unwrap()
        }];
        assert!(
            owned
                .reuse_for(&Request::new(Select::Features(&missing)))
                .is_none()
        );
    }

    /// The native value survives into the offer (a `u32`/`bool` feature isn't flattened to
    /// f32), and the borrowed-`Offer` bridge works for the lend-to-many-codecs path.
    #[test]
    fn offer_preserves_native_values_and_lends_borrowed() {
        let (w, h) = (32u32, 32u32);
        let query = AnalysisQuery::new(
            FeatureSet::just(AnalysisFeature::Variance).with(AnalysisFeature::PixelCount),
        );
        let owned = extract_offer(&img(w, h), w, h, &query, rgb8_srgb_descriptor_hash());

        // pixel_count is a u32 feature — preserved natively, projected on demand.
        let pc = owned.get("pixel_count").expect("pixel_count present");
        assert_eq!(pc.value(), Value::U32(w * h));
        assert_eq!(pc.float(), (w * h) as f32);

        // materialize one borrowed Offer and lend it (zero-cost for each codec after this).
        let cells: Vec<_> = owned
            .features()
            .iter()
            .map(OwnedFeatureResult::as_ref)
            .collect();
        let offer = Offer::new(&cells, owned.provenance());
        assert_eq!(offer.provenance().analyzer_version(), analyzer_version());
        assert_eq!(
            offer.get("variance").map(|f| f.value()),
            owned.get("variance").map(|f| f.value())
        );
    }
}
