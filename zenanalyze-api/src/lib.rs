//! # zenanalyze-api — the frozen feature contract for the picker crate tree
//!
//! A product links **many `zenanalyze` versions at once** — a dozen codecs each
//! pin the `zenanalyze`/`zenpredict` version their model was trained against,
//! and `zenanalyze0_2::*` ≠ `zenanalyze1_0::*` are incompatible types. So no
//! `zenanalyze` type can be the thing that crosses between layers. **This crate
//! is that thing**: the one type everyone agrees on, depended on at a single
//! version so it *unifies* across the whole build.
//!
//! It contains **only transport** — feature *names*, *values*, and a *reuse key*
//! — and never any feature definition, id, or extraction. That is exactly why it
//! can stay frozen: the feature math churns every `zenanalyze` release, but
//! `name → value + a reuse key + gather-by-name` does not.
//!
//! ## Why this crate is `1.x` and must stay `1.x` forever
//!
//! Cargo only unifies **semver-compatible** versions. All `1.*` unify to one
//! linked copy; the moment a `2.0` ships, the ecosystem splits into two
//! incompatible `zenanalyze-api` types and the whole point is lost. (A `0.x`
//! crate is *worse*: `0.1` and `0.2` already don't unify, so the first additive
//! release would split it.) Therefore: **this crate is `1.0`, evolves only
//! additively, and never goes to `2.0`.** [`Request`] and [`Offer`] are
//! `#[non_exhaustive]` with constructors precisely so new fields can be added
//! within `1.x` without breaking — every conceivable extension (owned offers,
//! typed values, per-feature metadata) is then a non-breaking addition.
//!
//! ## The reuse key: `(analyzer major.minor, defs_version, config_hash)`
//!
//! A feature is identified across versions by its **name**. Whether a *value* is
//! reusable is gated by a three-part key, because three independent things can
//! change a feature's number while its name stays fixed:
//!
//! 1. **`analyzer_version`** (`major.minor`) — the crate whose code computed it.
//!    A different *major.minor* may have changed the math; the patch is ignored.
//! 2. **`defs_version`** — within a `major.minor`, a bump signals the *numeric
//!    definition* of some feature changed (a `zenanalyze` compile-time const).
//! 3. **`config_hash`** — the value-affecting **runtime analysis config**
//!    (`zenanalyze::AnalysisQuery`: linear-light vs gamma today, more later).
//!    This is the subtle one: the same analyzer build (same `defs_version`)
//!    computes a *different* `variance` under `with_linear_light(true)` than
//!    under the gamma default, while the name and `defs_version` are identical.
//!    Without `config_hash`, a codec could reuse linear-light features against a
//!    gamma-trained model and feed it silently wrong inputs. `config_hash` is an
//!    **opaque digest** (`AnalysisQuery::config_hash()`, `0` = canonical
//!    default); equality is the only defined operation, so new config axes fold
//!    into the hash without ever touching this frozen crate.
//!
//! Source-dependent properties (primaries, transfer function, diffuse-white,
//! pixel layout) are *not* in the key on purpose: they're pinned by *which
//! image* you analyze, and you only ever reuse an [`Offer`] for the same image,
//! so every consumer gets that image's actual color properties.
//!
//! ## The flow it enables
//!
//! ```text
//! 1. each codec declares a Request (its model's feature names + its reuse key)
//! 2. the caller groups Requests by (defs_version, config_hash), unions the
//!    names in each group, picks the best zenanalyze it has, and runs ONE pass
//!    per group  ──▶  an Offer
//! 3. each codec: offer.reuse_for(my_request)?  Some(vec) => reuse  |  None => own pass
//! ```
//!
//! Grouping by `config_hash` is the caller's job; mixing configs into one pass
//! is not *unsound* (mismatches fall through [`Offer::reuse_for`] to `None` and
//! an own-pass), only a missed-reuse perf cost.
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
/// reuse key it needs them computed under.
///
/// A codec builds this from its baked model (the model carries its feature
/// columns + the version/config it was trained against). The caller collects a
/// `Request` from every codec — codecs on *different* `zenanalyze` versions, but
/// all producing this *same* `Request` type — and [`union_names`] the names in
/// each `(defs_version, config_hash)` group into a single pass.
///
/// `#[non_exhaustive]`: construct via [`Request::new`]; future fields are added
/// non-breakingly. Read fields directly.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Request<'a> {
    /// Feature column names the consumer wants. Names are the cross-version key
    /// (ids are version-local and must never cross a boundary).
    pub names: &'a [&'a str],
    /// The `zenanalyze` crate version the model was trained against, e.g.
    /// `"0.2.7"`. Its `major.minor` is part of the reuse key (patch ignored).
    pub analyzer_version: &'a str,
    /// The feature-definitions version (`zenanalyze::feature_defs_version()`) the
    /// model was trained against.
    pub defs_version: u32,
    /// The value-affecting analysis-config digest
    /// (`zenanalyze::AnalysisQuery::config_hash()`) the model was trained under.
    /// `0` = the canonical default config (gamma, no mode flags). Opaque —
    /// only equality is meaningful.
    pub config_hash: u64,
}

impl<'a> Request<'a> {
    /// Build a request from a model's feature names and its reuse key. Pass
    /// `config_hash = 0` for the default analysis config.
    #[must_use]
    pub const fn new(
        names: &'a [&'a str],
        analyzer_version: &'a str,
        defs_version: u32,
        config_hash: u64,
    ) -> Self {
        Self {
            names,
            analyzer_version,
            defs_version,
            config_hash,
        }
    }
}

/// A self-describing feature result: name→value pairs plus the reuse key of the
/// `zenanalyze` pass that produced them. Produced once by the caller and offered
/// to every codec, which decides whether it can reuse it.
///
/// `#[non_exhaustive]`: construct via [`Offer::new`]; future fields are added
/// non-breakingly. Read fields directly.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Offer<'a> {
    /// Feature names present in this offer, parallel to [`values`](Self::values).
    pub names: &'a [&'a str],
    /// Computed values, parallel to [`names`](Self::names). `f32` is the model
    /// input currency; every current feature (counts, bit-depths, bool flags) is
    /// exactly representable. A typed channel can be added additively if that
    /// ever stops holding.
    pub values: &'a [f32],
    /// The `zenanalyze` crate version that produced this offer.
    pub analyzer_version: &'a str,
    /// The feature-definitions version that produced this offer.
    pub defs_version: u32,
    /// The value-affecting analysis-config digest the pass ran under (`0` =
    /// canonical default). Must equal a consumer's [`Request::config_hash`] to be
    /// reusable.
    pub config_hash: u64,
}

impl<'a> Offer<'a> {
    /// Build an offer from a completed extraction. `names`/`values` are parallel;
    /// the reuse key is the pass's `(analyzer_version, defs_version, config_hash)`.
    #[must_use]
    pub const fn new(
        names: &'a [&'a str],
        values: &'a [f32],
        analyzer_version: &'a str,
        defs_version: u32,
        config_hash: u64,
    ) -> Self {
        Self {
            names,
            values,
            analyzer_version,
            defs_version,
            config_hash,
        }
    }

    /// Whether this offer was produced by feature definitions **and config**
    /// compatible with a consumer needing `(want_version, want_defs,
    /// want_config)`: same analyzer `major.minor` AND same `defs_version` AND
    /// same `config_hash`. The reuse gate — a `0.2` offer cannot satisfy a
    /// `1.0`-trained model (different math), within-major numeric drift
    /// (mismatched `defs_version`) is rejected, and a different analysis config
    /// (e.g. linear-light vs gamma `variance`) is rejected even at the same
    /// version.
    #[must_use]
    pub fn matches(&self, want_version: &str, want_defs: u32, want_config: u64) -> bool {
        self.defs_version == want_defs
            && self.config_hash == want_config
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
    /// the model a silent zero. Does NOT check the reuse key; pair with
    /// [`matches`](Self::matches) (or use [`reuse_for`](Self::reuse_for)).
    #[must_use]
    pub fn gather(&self, names: &[&str]) -> Option<Vec<f32>> {
        names.iter().map(|n| self.get(n)).collect()
    }

    /// Reuse this offer for `request` iff it is reuse-key-compatible AND carries
    /// every requested name: `Some(vector)` ready for the model, or `None` —
    /// run your own pass.
    #[must_use]
    pub fn reuse_for(&self, request: &Request<'_>) -> Option<Vec<f32>> {
        if self.matches(
            request.analyzer_version,
            request.defs_version,
            request.config_hash,
        ) {
            self.gather(request.names)
        } else {
            None
        }
    }
}

/// The distinct feature names across a set of requests — the name list a single
/// union pass should extract. First-seen order.
///
/// Call this **per `(defs_version, config_hash)` group**: one analyzer pass has
/// one version and one config, so unioning names across configs that differ in
/// value (e.g. linear-light vs gamma) would produce one number where two are
/// needed. Mismatches are caught safely by [`Offer::reuse_for`] (→ own-pass), so
/// mixing is a missed-reuse cost, not a correctness bug.
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

/// `"0.2.7"` → `"0.2"`. Patch (and any pre-release / build metadata after the
/// minor) don't change the reuse key — numeric drift within a minor is what
/// `defs_version` catches. Degenerate inputs (`"1"`, `""`) are returned as-is.
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
        Offer::new(OFFER_NAMES, OFFER_VALS, "0.2.7", 1, 0)
    }

    #[test]
    fn matches_is_major_minor_plus_defs_plus_config() {
        let o = offer();
        assert!(o.matches("0.2.7", 1, 0)); // exact
        assert!(o.matches("0.2.3", 1, 0)); // patch differs, reuse key same
        assert!(!o.matches("0.2.7", 2, 0)); // defs drift rejected
        assert!(!o.matches("0.2.7", 1, 99)); // config drift (e.g. linear-light) rejected
        assert!(!o.matches("1.0.0", 1, 0)); // different major rejected
        assert!(!o.matches("0.3.0", 1, 0)); // different minor rejected
    }

    #[test]
    fn major_minor_handles_degenerate_and_prerelease() {
        assert_eq!(major_minor("0.2.7"), "0.2");
        assert_eq!(major_minor("10.20.30"), "10.20");
        assert_eq!(major_minor("1.0.0-rc.1"), "1.0"); // pre-release after minor ignored
        assert_eq!(major_minor("1.0.0-beta.2+build.5"), "1.0");
        assert_eq!(major_minor("0.2"), "0.2"); // no patch
        assert_eq!(major_minor("1"), "1"); // no minor
        assert_eq!(major_minor(""), "");
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
    fn reuse_for_gates_on_key_then_coverage() {
        let o = offer();
        let want = ["edge_density", "variance"];
        // compatible key + covered => reuse
        assert_eq!(
            o.reuse_for(&Request::new(&want, "0.2.0", 1, 0)),
            Some(alloc::vec![2.0, 1.0])
        );
        // incompatible version => own pass even though names are present
        assert_eq!(o.reuse_for(&Request::new(&want, "1.0.0", 1, 0)), None);
        // incompatible config (linear-light vs gamma) => own pass, same names/version
        assert_eq!(o.reuse_for(&Request::new(&want, "0.2.0", 1, 42)), None);
        // compatible but a name missing => own pass
        assert_eq!(
            o.reuse_for(&Request::new(&["variance", "noise_floor_y"], "0.2.0", 1, 0)),
            None
        );
    }

    #[test]
    fn union_dedups_first_seen() {
        let a = Request::new(&["variance", "edge_density"], "0.2.0", 1, 0);
        let b = Request::new(&["edge_density", "uniformity"], "1.0.0", 1, 0);
        assert_eq!(
            union_names(&[a, b]),
            alloc::vec!["variance", "edge_density", "uniformity"]
        );
    }
}
