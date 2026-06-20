//! Producing a [`zenanalyze_api::Offer`] from an analysis pass — the producer
//! side of the version-unifying feature contract (gated by the `api` feature).
//!
//! [`OwnedOffer`] owns the dense `(names, values)` an [`Offer`] borrows, plus the
//! reuse key, and lends a borrowed [`Offer`] via [`OwnedOffer::as_offer`] — the
//! `PathBuf`→`&Path` relationship. The orchestrator runs one pass per
//! `(defs_version, config_hash)` group and offers it to every codec in the group;
//! each codec calls [`Offer::reuse_for`] to reuse or fall back to its own pass.
//!
//! The owned holder lives **here**, in the version-specific impl crate, not in
//! the frozen `zenanalyze-api` — so the contract stays a minimal borrowed type
//! and the storage that backs it is the producer's concern.

use crate::feature::AnalysisQuery;
use crate::{analyze_features_rgb8, analyzer_version, feature_defs_version, feature_name};
use zenanalyze_api::Offer;

/// Serialize the **extraction-level** provenance for a `query` + input framing —
/// the stamp to store alongside a feature table (e.g. one
/// `extract_features_for_picker` output, where every row shares this `query`,
/// config, and descriptor). Covers every feature in `query.features()` that has a
/// golden version row; the rest are omitted (unversioned → a consumer treats them
/// as not-reusable, the safe direction).
///
/// `descriptor_hash` is the value-affecting input framing
/// ([`crate::versioning::descriptor_hash`] / `descriptor_hash_of`, or
/// [`crate::versioning::rgb8_srgb_descriptor_hash`] for the RGB8 fast path). See
/// [`OwnedOffer::provenance`] for the per-offer variant that stamps exactly the
/// features one offer carries.
#[must_use]
pub fn feature_set_provenance(query: &AnalysisQuery, descriptor_hash: u64) -> String {
    let feats: Vec<(&str, u64)> = query
        .features()
        .iter()
        .filter_map(|feat| {
            feature_name(feat.id()).zip(crate::versioning::feature_version_hash(feat))
        })
        .collect();
    zenanalyze_api::provenance::write_provenance(
        analyzer_version(),
        query.config_hash(),
        descriptor_hash,
        &feats,
    )
}

/// An owned bundle of feature names + values + reuse key, backing a borrowed
/// [`Offer`]. Produce one with [`OwnedOffer::extract`], then lend it as an
/// [`Offer`] with [`as_offer`](Self::as_offer).
#[derive(Clone, Debug)]
pub struct OwnedOffer {
    names: Vec<&'static str>,
    values: Vec<f32>,
    config_hash: u64,
}

impl OwnedOffer {
    /// Run **one** analysis pass for `query` over an RGB8 buffer and bundle the
    /// result as an offerable unit.
    ///
    /// The offer carries exactly the features in `query.features()` that the pass
    /// computed — under the query's config (e.g. gamma vs `with_linear_light`) —
    /// paired with their canonical [`feature_name`]s, plus the reuse key
    /// `(analyzer_version, feature_defs_version, query.config_hash())`. Features
    /// the query requested but the build didn't compute (e.g. HDR features
    /// without the `hdr` feature) are simply absent from the offer, so a consumer
    /// that needs them gets `None` from [`Offer::reuse_for`] and runs its own pass
    /// rather than a silent zero.
    ///
    /// `rgb.len()` must be `width * height * 3` (same contract as
    /// [`analyze_features_rgb8`]).
    #[must_use]
    pub fn extract(rgb: &[u8], width: u32, height: u32, query: &AnalysisQuery) -> Self {
        let results = analyze_features_rgb8(rgb, width, height, query);
        let mut names = Vec::new();
        let mut values = Vec::new();
        for feat in query.features().iter() {
            if let (Some(name), Some(v)) = (feature_name(feat.id()), results.get_f32(feat)) {
                names.push(name);
                values.push(v);
            }
        }
        Self {
            names,
            values,
            config_hash: query.config_hash(),
        }
    }

    /// Borrow this bundle as a [`zenanalyze_api::Offer`] — the form a codec
    /// negotiates with [`Offer::reuse_for`].
    #[must_use]
    pub fn as_offer(&self) -> Offer<'_> {
        Offer::new(
            &self.names,
            &self.values,
            analyzer_version(),
            feature_defs_version(),
            self.config_hash,
        )
    }

    /// The feature names this bundle carries (parallel to [`values`](Self::values)).
    #[must_use]
    pub fn names(&self) -> &[&'static str] {
        &self.names
    }

    /// The feature values this bundle carries (parallel to [`names`](Self::names)).
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The analysis-config digest the pass ran under (`0` = canonical default).
    #[must_use]
    pub fn config_hash(&self) -> u64 {
        self.config_hash
    }

    /// Serialize this offer's **provenance** for storing alongside its values on
    /// disk (e.g. in Parquet key-value metadata), so a training run years later
    /// can validate reuse feature-by-feature. Records `(analyzer_version,
    /// config_hash, descriptor_hash)` plus each carried feature's
    /// [`feature_version_hash`](crate::versioning::feature_version_hash) — the
    /// three legs of the reuse key (code / config / input framing).
    ///
    /// `descriptor_hash` is the value-affecting input framing
    /// ([`crate::versioning::descriptor_hash`] from the analyzed slice, or
    /// [`descriptor_hash_of`](crate::versioning::descriptor_hash_of) for an
    /// explicit descriptor — `descriptor_hash_of(&PixelDescriptor::RGB8_SRGB,
    /// None)` for the RGB8 fast path). Features with no golden version row are
    /// omitted (unversioned → a consumer treats them as not-reusable, the safe
    /// direction), which by the `every_present_feature_has_a_version_hash` golden
    /// test means every carried feature is stamped.
    #[must_use]
    pub fn provenance(&self, descriptor_hash: u64) -> String {
        let feats: Vec<(&str, u64)> = self
            .names
            .iter()
            .filter_map(|&n| crate::versioning::feature_version_hash_by_name(n).map(|h| (n, h)))
            .collect();
        zenanalyze_api::provenance::write_provenance(
            analyzer_version(),
            self.config_hash,
            descriptor_hash,
            &feats,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{AnalysisFeature, FeatureSet};
    use zenanalyze_api::Request;

    /// The whole contract, end to end: zenanalyze runs a pass, lends an `Offer`,
    /// and a consumer's `Request` reuses it (or falls back) exactly as the reuse
    /// key dictates. This is the real-use validation the frozen shape needed.
    #[test]
    fn owned_offer_round_trips_through_reuse_for() {
        let (w, h) = (64u32, 64u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let query = AnalysisQuery::new(
            FeatureSet::just(AnalysisFeature::Variance).with(AnalysisFeature::EdgeDensity),
        );

        let owned = OwnedOffer::extract(&rgb, w, h, &query);
        let offer = owned.as_offer();
        assert_eq!(owned.config_hash(), 0, "gamma default config hashes to 0");
        assert_eq!(owned.names().len(), owned.values().len());

        // A consumer requesting a subset, in a different order, at the SAME
        // version + config: reuse, with order preserved.
        let names = ["edge_density", "variance"];
        let req = Request::new(&names, analyzer_version(), feature_defs_version(), 0);
        let v = offer
            .reuse_for(&req)
            .expect("same version+config+coverage must reuse");
        assert_eq!(v.len(), 2);
        assert_eq!(offer.get("edge_density"), Some(v[0]));
        assert_eq!(offer.get("variance"), Some(v[1]));

        // Same names + version, but a different analysis config (pretend
        // linear-light): must NOT reuse — this is the hole config_hash closes.
        let req_other_cfg =
            Request::new(&names, analyzer_version(), feature_defs_version(), 0xDEAD);
        assert!(
            offer.reuse_for(&req_other_cfg).is_none(),
            "config mismatch must force an own-pass"
        );

        // A name the offer doesn't carry: own-pass, never a silent zero.
        let req_missing = Request::new(
            &["variance", "noise_floor_y"],
            analyzer_version(),
            feature_defs_version(),
            0,
        );
        assert!(offer.reuse_for(&req_missing).is_none());
    }

    /// The serialization contract end to end: extract → stamp provenance → parse
    /// it back → confirm it validates each feature against the live analyzer, and
    /// rejects a different input framing (descriptor).
    #[test]
    fn provenance_round_trips_and_gates_on_descriptor() {
        use zenanalyze_api::provenance::OwnedProvenance;
        use zenpixels::PixelDescriptor;

        let (w, h) = (32u32, 32u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let query = AnalysisQuery::new(
            FeatureSet::just(AnalysisFeature::Variance).with(AnalysisFeature::EdgeDensity),
        );
        let owned = OwnedOffer::extract(&rgb, w, h, &query);

        // RGB8 fast path ⇒ the sRGB descriptor framing.
        let dh = crate::versioning::descriptor_hash_of(&PixelDescriptor::RGB8_SRGB, None);
        let text = owned.provenance(dh);

        let prov = OwnedProvenance::parse(&text).expect("our own provenance must parse");
        assert_eq!(prov.config_hash(), owned.config_hash());
        assert_eq!(prov.descriptor_hash(), dh);
        assert_eq!(prov.analyzer_version(), analyzer_version());

        // Every carried feature validates against the live analyzer under the
        // same (code, config, framing) — the whole point of the stamp.
        for &name in owned.names() {
            let live = crate::versioning::feature_version_hash_by_name(name)
                .expect("a carried feature is versioned (golden invariant)");
            assert!(
                prov.feature_is_reusable(name, live, owned.config_hash(), dh),
                "{name} should be reusable under the matching key"
            );
            // A different input framing (e.g. a Display-P3 PQ descriptor) must
            // fall out — same pixels, different values.
            assert!(
                !prov.feature_is_reusable(name, live, owned.config_hash(), dh ^ 0x1),
                "{name} must NOT reuse across a descriptor mismatch"
            );
        }
    }
}
