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

use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};
use crate::versioning::{feature_version_hash, rgb8_srgb_descriptor_hash};
use crate::{AnalyzeError, analyze_features_rgb8, analyzer_version, feature_name};
use zenanalyze_api::{
    NamedFeature, OwnedFeatureResult, OwnedOffer, Provenance, Request, Select, Value,
};

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

/// Resolve one bare feature name to a feature this build can actually extract.
/// An error rather than a silent drop: a caller that asked for a column must get it or
/// be told this build cannot produce it.
fn supported_feature(name: &str) -> Result<AnalysisFeature, AnalyzeError> {
    let f = AnalysisFeature::from_name(name)
        .filter(|f| FeatureSet::SUPPORTED.contains(*f))
        .ok_or_else(|| {
            AnalyzeError::InvalidInput(format!(
                "this zenanalyze build cannot produce feature {name:?}"
            ))
        })?;
    Ok(f)
}

/// Run one analysis pass answering a [`Request`] — the "my offer wasn't enough, scan it
/// myself" path, in one call.
///
/// This is the **own-scan** half of the contract's flow. A codec first asks the offer the
/// host gave it (`offer.satisfies(&req)` / `reuse_for`); when the answer is no, it calls
/// this with the same [`Request`] and gets an [`OwnedOffer`] it can negotiate against
/// identically. The `Request` → [`FeatureSet`] resolution lives here, once, so no consumer
/// re-implements it.
///
/// The returned offer is stamped with this build's [`Provenance`], exactly as
/// [`extract_offer`] does — so a value that came from here is indistinguishable in kind
/// from one the host produced, and the two blend under the same `schema_hash` gate.
///
/// # Errors
///
/// - [`AnalyzeError::InvalidInput`] if `rgb.len() != width * height * 3`, a dimension is
///   zero, the dimensions overflow, or the request names a feature this build cannot
///   produce — including a [`Select::Features`] want whose code version does not match
///   this build's (a version-pinned want MUST miss on a drift rather than silently take
///   the local value; that miss is the safety property `Select::Features` exists for).
/// - [`AnalyzeError::InvalidInput`] for a [`Select`] variant added to the contract after
///   this build: `Select` is `#[non_exhaustive]`, and guessing at an unknown selector
///   would violate "produce what was asked or say you cannot".
///
/// ```no_run
/// # #[cfg(feature = "api")] {
/// use zenanalyze_api::{Request, Select};
///
/// let wants = ["variance", "edge_density"];
/// let offer = zenanalyze::offer_for_request(
///     &[0u8; 8 * 8 * 3], 8, 8, &Request::new(Select::Names(&wants)),
/// ).unwrap();
/// assert!(offer.get("variance").is_some());
/// # }
/// ```
pub fn offer_for_request(
    rgb: &[u8],
    width: u32,
    height: u32,
    request: &Request<'_>,
) -> Result<OwnedOffer, AnalyzeError> {
    // Resolve the ask to a FeatureSet BEFORE touching pixels — an unmeetable want is an
    // error, never a pass whose offer quietly lacks a column.
    let set = match request.select() {
        Select::All => FeatureSet::SUPPORTED,
        Select::Features(wants) => {
            let mut set = FeatureSet::new();
            for want in wants {
                let f = supported_feature(want.name())?;
                // A version-pinned want must match THIS build's code version exactly.
                let full = feature_version_hash(f).ok_or_else(|| {
                    AnalyzeError::InvalidInput(format!(
                        "feature {:?} has no golden version row in this build",
                        want.name()
                    ))
                })?;
                if NamedFeature::fold_hash(full) != want.version_hash() {
                    return Err(AnalyzeError::InvalidInput(format!(
                        "feature {:?} drifted: this build is a different code version than \
                         the pinned want",
                        want.name()
                    )));
                }
                set = set.with(f);
            }
            set
        }
        Select::Names(names) => {
            let mut set = FeatureSet::new();
            for name in names {
                set = set.with(supported_feature(name)?);
            }
            set
        }
        // `Select` is `#[non_exhaustive]`: a selector added to the contract after this
        // build can't be honored, and guessing would violate "produce it or say so".
        _ => {
            return Err(AnalyzeError::InvalidInput(
                "this zenanalyze build does not understand the requested Select variant".into(),
            ));
        }
    };

    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| AnalyzeError::InvalidInput("dimensions overflow".into()))?;
    if width == 0 || height == 0 || rgb.len() != expected {
        return Err(AnalyzeError::InvalidInput(format!(
            "rgb.len() {} != width {width} * height {height} * 3",
            rgb.len()
        )));
    }

    let query = AnalysisQuery::new(set);
    Ok(extract_offer(
        rgb,
        width,
        height,
        &query,
        rgb8_srgb_descriptor_hash(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::versioning::feature_qualified_names;
    use zenanalyze_api::{Offer, Value};

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

    /// The own-scan path: `offer_for_request` answers a contract `Request` directly, so a
    /// codec whose handed-down offer wasn't enough re-scans with the SAME request object.
    #[test]
    fn offer_for_request_serves_the_own_scan_path() {
        let (w, h) = (32u32, 32u32);

        // Version-agnostic ask (a threshold heuristic) — resolves and reuses.
        let names = ["variance", "edge_density"];
        let by_name = Request::new(Select::Names(&names));
        let offer = offer_for_request(&img(w, h), w, h, &by_name).expect("supported names extract");
        assert_eq!(offer.provenance().analyzer_version(), analyzer_version());
        assert!(
            offer.satisfies(&by_name),
            "the own scan must cover its own ask"
        );
        assert_eq!(offer.reuse_for(&by_name).map(|v| v.len()), Some(2));

        // Version-pinned ask built from THIS build's own qualified names — reuses too.
        let pinned: Vec<_> = feature_qualified_names()
            .into_iter()
            .filter(|(_, q)| q.starts_with("variance@"))
            .map(|(_, q)| q)
            .collect();
        assert_eq!(pinned.len(), 1);
        let pinned_named = [NamedFeature::parse(&pinned[0]).unwrap()];
        let pinned_req = Request::new(Select::Features(&pinned_named));
        let pinned_offer = offer_for_request(&img(w, h), w, h, &pinned_req)
            .expect("our own qualified name must be extractable");
        assert!(pinned_offer.satisfies(&pinned_req));
    }

    /// An unmeetable ask and a malformed buffer are both errors — never an offer that
    /// quietly lacks a requested column.
    #[test]
    fn offer_for_request_rejects_unmeetable_asks_and_bad_buffers() {
        let (w, h) = (16u32, 16u32);

        let unknown = ["definitely_not_a_feature"];
        assert!(
            offer_for_request(&img(w, h), w, h, &Request::new(Select::Names(&unknown))).is_err()
        );

        // Right name, wrong code version ⇒ a pinned want this build cannot honor. This miss
        // is the whole safety property of `Select::Features`.
        let drift = [NamedFeature::parse("variance@ffffffff").unwrap()];
        assert!(
            offer_for_request(&img(w, h), w, h, &Request::new(Select::Features(&drift))).is_err()
        );

        let names = ["variance"];
        let req = Request::new(Select::Names(&names));
        for (buf, bw, bh) in [(img(w, h), w, h + 1), (img(w, h), 0, h)] {
            assert!(offer_for_request(&buf, bw, bh, &req).is_err());
        }
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
