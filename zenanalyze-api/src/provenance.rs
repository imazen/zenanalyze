//! Feature **serialization provenance** — the dep-free contract for recording how
//! a serialized feature set was produced, so it can be validated for reuse years
//! later (training data outlives the analyzer that made it).
//!
//! A serialized feature vector is reproducible/reusable only under the same
//! *(code, config, input framing)*. This module captures all three as a compact,
//! human-readable text block you store alongside the values (e.g. in a Parquet
//! key-value metadata entry):
//!
//! * **`analyzer_version`** — informational (the version hash already encodes the
//!   caret-compatibility root).
//! * **`config_hash`** — the value-affecting `AnalysisQuery` config
//!   (`AnalysisQuery::config_hash()`; gamma vs linear-light, …).
//! * **`descriptor_hash`** — the value-affecting input framing (primaries /
//!   transfer / alpha / diffuse-white). The *same pixels* under different
//!   primaries or transfer produce different features, so this is part of the
//!   reuse key, not just the pixels.
//! * **per feature: `name = version_hash`** — `zenanalyze::feature_version_hash`,
//!   so a future run checks compatibility **feature-by-feature** (only the changed
//!   features fall out; the rest stay reusable).
//!
//! ## Format (`zenanalyze-provenance/1`)
//!
//! ```text
//! zenanalyze-provenance/1
//! analyzer_version=0.2.0
//! config_hash=0
//! descriptor_hash=81985529216486895
//! [features]
//! variance=12997637936314813
//! edge_density=4733065156644888998
//! ```
//!
//! Pure `core` + `alloc`, no dependencies, so it never forces a version split.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const MAGIC: &str = "zenanalyze-provenance/1";

/// Serialize a feature set's provenance to the `zenanalyze-provenance/1` text
/// block. `features` is `(name, version_hash)` in any order (written verbatim).
#[must_use]
pub fn write_provenance(
    analyzer_version: &str,
    config_hash: u64,
    descriptor_hash: u64,
    features: &[(&str, u64)],
) -> String {
    let mut s = String::new();
    s.push_str(MAGIC);
    s.push('\n');
    s.push_str(&format!("analyzer_version={analyzer_version}\n"));
    s.push_str(&format!("config_hash={config_hash}\n"));
    s.push_str(&format!("descriptor_hash={descriptor_hash}\n"));
    s.push_str("[features]\n");
    for (name, hash) in features {
        s.push_str(&format!("{name}={hash}\n"));
    }
    s
}

/// Why a provenance block could not be parsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProvenanceError {
    /// The first line was not a recognized `zenanalyze-provenance/<v>` magic.
    UnknownFormat,
    /// A required header (`analyzer_version` / `config_hash` / `descriptor_hash`)
    /// was missing.
    MissingHeader,
    /// A `key=value` / `name=hash` line was malformed, or a hash didn't parse.
    BadLine,
}

/// A parsed, owned provenance block. Build with [`OwnedProvenance::parse`], query
/// with [`feature_hash`](Self::feature_hash); compare its `config_hash` /
/// `descriptor_hash` and each feature's hash against the current analyzer to
/// decide, per feature, whether a serialized value is reusable.
#[derive(Clone, Debug)]
pub struct OwnedProvenance {
    analyzer_version: String,
    config_hash: u64,
    descriptor_hash: u64,
    features: Vec<(String, u64)>,
}

impl OwnedProvenance {
    /// Parse a `zenanalyze-provenance/1` block. Forward-compatible on unknown
    /// header keys (ignored) but strict on the three required headers + the magic.
    pub fn parse(text: &str) -> Result<Self, ProvenanceError> {
        let mut lines = text.lines();
        match lines.next() {
            Some(l) if l.trim() == MAGIC => {}
            _ => return Err(ProvenanceError::UnknownFormat),
        }
        let mut analyzer_version: Option<String> = None;
        let mut config_hash: Option<u64> = None;
        let mut descriptor_hash: Option<u64> = None;
        let mut features = Vec::new();
        let mut in_features = false;
        for raw in lines {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if line == "[features]" {
                in_features = true;
                continue;
            }
            let (key, val) = line.split_once('=').ok_or(ProvenanceError::BadLine)?;
            let (key, val) = (key.trim(), val.trim());
            if in_features {
                let hash = val.parse().map_err(|_| ProvenanceError::BadLine)?;
                features.push((key.to_string(), hash));
            } else {
                match key {
                    "analyzer_version" => analyzer_version = Some(val.to_string()),
                    "config_hash" => {
                        config_hash = Some(val.parse().map_err(|_| ProvenanceError::BadLine)?)
                    }
                    "descriptor_hash" => {
                        descriptor_hash = Some(val.parse().map_err(|_| ProvenanceError::BadLine)?)
                    }
                    _ => {} // forward-compatible: ignore unknown headers
                }
            }
        }
        Ok(Self {
            analyzer_version: analyzer_version.ok_or(ProvenanceError::MissingHeader)?,
            config_hash: config_hash.ok_or(ProvenanceError::MissingHeader)?,
            descriptor_hash: descriptor_hash.ok_or(ProvenanceError::MissingHeader)?,
            features,
        })
    }

    /// The `zenanalyze` version that produced these features (informational).
    #[must_use]
    pub fn analyzer_version(&self) -> &str {
        &self.analyzer_version
    }
    /// The `AnalysisQuery::config_hash()` the features were extracted under.
    #[must_use]
    pub fn config_hash(&self) -> u64 {
        self.config_hash
    }
    /// The input-framing (primaries/transfer/alpha/diffuse-white) hash.
    #[must_use]
    pub fn descriptor_hash(&self) -> u64 {
        self.descriptor_hash
    }
    /// The recorded `feature_version_hash` for `name`, or `None` if absent.
    #[must_use]
    pub fn feature_hash(&self, name: &str) -> Option<u64> {
        self.features
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, h)| *h)
    }
    /// All `(name, version_hash)` pairs, in file order.
    #[must_use]
    pub fn features(&self) -> &[(String, u64)] {
        &self.features
    }

    /// Whether the serialized `name` is reusable by an analyzer that would now
    /// compute it with `current_version_hash`, under `current_config_hash` +
    /// `current_descriptor_hash`. All three must match: the feature's code
    /// version, the analysis config, and the input framing.
    #[must_use]
    pub fn feature_is_reusable(
        &self,
        name: &str,
        current_version_hash: u64,
        current_config_hash: u64,
        current_descriptor_hash: u64,
    ) -> bool {
        self.config_hash == current_config_hash
            && self.descriptor_hash == current_descriptor_hash
            && self.feature_hash(name) == Some(current_version_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let feats = [("variance", 111u64), ("edge_density", 222u64)];
        let text = write_provenance("0.2.0", 0, 81985529216486895, &feats);
        let p = OwnedProvenance::parse(&text).expect("parse");
        assert_eq!(p.analyzer_version(), "0.2.0");
        assert_eq!(p.config_hash(), 0);
        assert_eq!(p.descriptor_hash(), 81985529216486895);
        assert_eq!(p.feature_hash("variance"), Some(111));
        assert_eq!(p.feature_hash("edge_density"), Some(222));
        assert_eq!(p.feature_hash("absent"), None);
    }

    #[test]
    fn reuse_gate_needs_all_three() {
        let text = write_provenance("0.2.0", 0, 9, &[("variance", 111u64)]);
        let p = OwnedProvenance::parse(&text).unwrap();
        assert!(p.feature_is_reusable("variance", 111, 0, 9)); // all match
        assert!(!p.feature_is_reusable("variance", 999, 0, 9)); // code drifted
        assert!(!p.feature_is_reusable("variance", 111, 5, 9)); // config differs (linear vs gamma)
        assert!(!p.feature_is_reusable("variance", 111, 0, 7)); // descriptor differs (P3 vs sRGB)
        assert!(!p.feature_is_reusable("absent", 111, 0, 9)); // not recorded
    }

    #[test]
    fn rejects_unknown_format_and_missing_headers() {
        assert_eq!(
            OwnedProvenance::parse("garbage").unwrap_err(),
            ProvenanceError::UnknownFormat
        );
        assert_eq!(
            OwnedProvenance::parse("zenanalyze-provenance/1\nconfig_hash=0\n").unwrap_err(),
            ProvenanceError::MissingHeader
        );
    }

    #[test]
    fn ignores_unknown_headers_forward_compatibly() {
        let text = "zenanalyze-provenance/1\nanalyzer_version=0.2.0\nconfig_hash=0\n\
                    descriptor_hash=9\nfuture_field=whatever\n[features]\nvariance=1\n";
        let p = OwnedProvenance::parse(text).expect("unknown headers are ignored");
        assert_eq!(p.feature_hash("variance"), Some(1));
    }
}
