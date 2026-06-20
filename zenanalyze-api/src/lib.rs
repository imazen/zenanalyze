//! # zenanalyze-api — the frozen feature contract for the picker crate tree
//!
//! A product links **many `zenanalyze` versions at once** — a dozen codecs each
//! pin the `zenanalyze`/`zenpredict` version their model was trained against,
//! and `zenanalyze0_2::*` ≠ `zenanalyze1_0::*` are incompatible types. So no
//! `zenanalyze` type can be the thing that crosses between layers. **This crate
//! is that thing**: the one type everyone agrees on, depended on at a single
//! version so it *unifies* across the whole build.
//!
//! It contains **only transport** — feature *names*, *values*, and *version
//! stamps* — and never any feature definition, id, or extraction. That is
//! exactly why it can stay frozen: the feature math churns every `zenanalyze`
//! release, but `name → value + a version stamp + gather-by-name` does not.
//!
//! ## The flow it enables
//!
//! ```text
//! 1. each codec declares a Request (its model's feature names + the version it needs)
//! 2. the caller unions all Requests, picks the best zenanalyze it has,
//! 3. runs ONE zenanalyze pass over the union  ──▶  an Offer
//! 4. each codec: offer.reuse_for(my_request)?  Some(vec) => reuse  |  None => own pass
//! ```
//!
//! `zenanalyze@X` (the multi-version impl) depends on this crate and produces an
//! [`Offer`] from an extraction; consumers ([`zenpredict`]-based pickers) depend
//! on this crate to negotiate one. Inference still flows as a plain `&[f32]`.
//!
//! [`zenpredict`]: https://crates.io/crates/zenpredict

#![no_std]

extern crate alloc;
use alloc::vec::Vec;

/// What a consumer wants: the feature column **names** its model needs, and the
/// `zenanalyze` version it needs them computed at.
///
/// A codec builds this from its baked model (the model carries its feature
/// columns + the version it was trained against). The caller collects a
/// `Request` from every codec — codecs on *different* `zenanalyze` versions, but
/// all producing this *same* `Request` type — and [`union_names`] them into the
/// single pass.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Feature column names the consumer wants. Names are the cross-version key
    /// (ids are version-local and must never cross a boundary).
    pub names: &'a [&'a str],
    /// The `zenanalyze` crate version the model was trained against, e.g.
    /// `"0.2.7"`. Its `major.minor` + [`defs_version`](Self::defs_version) is the
    /// reuse key.
    pub analyzer_version: &'a str,
    /// The feature-definitions version (`zenanalyze::feature_defs_version()`) the
    /// model was trained against.
    pub defs_version: u32,
}

/// A self-describing feature result: name→value pairs plus the version stamp of
/// the `zenanalyze` pass that produced them. Produced once by the caller and
/// offered to every codec, which decides whether it can reuse it.
#[derive(Clone, Copy, Debug)]
pub struct Offer<'a> {
    /// Feature names present in this offer, parallel to [`values`](Self::values).
    pub names: &'a [&'a str],
    /// Computed values, parallel to [`names`](Self::names).
    pub values: &'a [f32],
    /// The `zenanalyze` crate version that produced this offer.
    pub analyzer_version: &'a str,
    /// The feature-definitions version that produced this offer.
    pub defs_version: u32,
}

impl<'a> Offer<'a> {
    /// Whether this offer was produced by feature definitions **compatible** with
    /// a consumer needing `(want_version, want_defs)`: same analyzer
    /// `major.minor` AND same `defs_version`. The reuse gate — a `0.2` offer
    /// cannot satisfy a `1.0`-trained model (different math), and within-major
    /// numeric drift (mismatched `defs_version`) is rejected.
    #[must_use]
    pub fn matches(&self, want_version: &str, want_defs: u32) -> bool {
        self.defs_version == want_defs
            && major_minor(self.analyzer_version) == major_minor(want_version)
    }

    /// The value for `name`, or `None` if this offer doesn't carry it.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f32> {
        self.names
            .iter()
            .position(|n| *n == name)
            .map(|i| self.values[i])
    }

    /// Gather `names` into a model-input vector **in the given order**. `None` if
    /// any name is absent — the consumer then runs its own pass rather than feed
    /// the model a silent zero. Does NOT check the version; pair with
    /// [`matches`](Self::matches) (or use [`reuse_for`](Self::reuse_for)).
    #[must_use]
    pub fn gather(&self, names: &[&str]) -> Option<Vec<f32>> {
        names.iter().map(|n| self.get(n)).collect()
    }

    /// Reuse this offer for `request` iff it is version-compatible AND carries
    /// every requested name: `Some(vector)` ready for the model, or `None` —
    /// run your own pass.
    #[must_use]
    pub fn reuse_for(&self, request: &Request<'_>) -> Option<Vec<f32>> {
        if self.matches(request.analyzer_version, request.defs_version) {
            self.gather(request.names)
        } else {
            None
        }
    }
}

/// The distinct feature names across a set of requests — the name list the
/// caller's single union pass should extract. First-seen order.
#[must_use]
pub fn union_names<'a>(requests: &[Request<'a>]) -> Vec<&'a str> {
    let mut out: Vec<&'a str> = Vec::new();
    for r in requests {
        for &n in r.names {
            if !out.contains(&n) {
                out.push(n);
            }
        }
    }
    out
}

/// `"0.2.7"` → `"0.2"`. Patch differences don't change the reuse key (numeric
/// drift within a minor is what `defs_version` catches).
fn major_minor(v: &str) -> &str {
    match v.match_indices('.').nth(1) {
        Some((i, _)) => &v[..i],
        None => v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFER_NAMES: &[&str] = &["variance", "edge_density", "uniformity"];
    const OFFER_VALS: &[f32] = &[1.0, 2.0, 3.0];

    fn offer() -> Offer<'static> {
        Offer {
            names: OFFER_NAMES,
            values: OFFER_VALS,
            analyzer_version: "0.2.7",
            defs_version: 1,
        }
    }

    #[test]
    fn matches_is_major_minor_plus_defs() {
        let o = offer();
        assert!(o.matches("0.2.7", 1)); // exact
        assert!(o.matches("0.2.3", 1)); // patch differs, reuse key same
        assert!(!o.matches("0.2.7", 2)); // defs drift rejected
        assert!(!o.matches("1.0.0", 1)); // different major rejected
        assert!(!o.matches("0.3.0", 1)); // different minor rejected
    }

    #[test]
    fn gather_preserves_order_or_fails() {
        let o = offer();
        assert_eq!(
            o.gather(&["uniformity", "variance"]),
            Some(alloc::vec![3.0, 1.0])
        );
        assert_eq!(o.gather(&["variance", "absent"]), None); // never a silent zero
    }

    #[test]
    fn reuse_for_gates_on_version_then_coverage() {
        let o = offer();
        let want = ["edge_density", "variance"];
        // compatible + covered => reuse
        assert_eq!(
            o.reuse_for(&Request {
                names: &want,
                analyzer_version: "0.2.0",
                defs_version: 1
            }),
            Some(alloc::vec![2.0, 1.0])
        );
        // incompatible version => own pass even though names are present
        assert_eq!(
            o.reuse_for(&Request {
                names: &want,
                analyzer_version: "1.0.0",
                defs_version: 1
            }),
            None
        );
        // compatible but a name missing => own pass
        assert_eq!(
            o.reuse_for(&Request {
                names: &["variance", "noise_floor_y"],
                analyzer_version: "0.2.0",
                defs_version: 1,
            }),
            None
        );
    }

    #[test]
    fn union_dedups_first_seen() {
        let a = Request {
            names: &["variance", "edge_density"],
            analyzer_version: "0.2.0",
            defs_version: 1,
        };
        let b = Request {
            names: &["edge_density", "uniformity"],
            analyzer_version: "1.0.0",
            defs_version: 1,
        };
        assert_eq!(
            union_names(&[a, b]),
            alloc::vec!["variance", "edge_density", "uniformity"]
        );
    }
}
