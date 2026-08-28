//! The crate-level documentation is the README, included verbatim so its
//! examples are compiled & run as doctests (`cargo test`) — drift between the
//! docs and the API fails the build. Per-item rustdoc is on each item below.
#![doc = include_str!("../README.md")]
#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

// ───────────────────────────── NamedFeature ────────────────────────────────

/// A feature's cross-version **identity**, carried as its self-describing qualified name
/// `"variance@b4a1c2d3"` (`name` `@` 8 lowercase hex of the 32-bit
/// [`version_hash`](Self::version_hash)). **One borrowed `&str`** — the qualified name IS
/// the identity, so equality, hashing, and reuse are a single string compare; the `name`
/// and `version_hash` are zero-alloc on-demand splits used only on cold diagnostic paths.
///
/// The version hash is the feature's **code** version ([`fold_hash`](Self::fold_hash) of
/// `zenanalyze::feature_version_hash`) — build-stable, so the qualified name is too. It does
/// NOT fold in the runtime analysis config or input framing: those are constant within one
/// analysis pass (one image), so reuse against that pass's [`Offer`] is safe by code-version
/// alone, and they instead travel on the per-offer [`Provenance`] (for debugging and the
/// serialization blend gate). Keeping the name build-stable is also what lets a producer
/// intern it once and lend it cheaply.
///
/// Construct from a validated string with [`parse`](Self::parse) / `TryFrom` (both also
/// build a **compile-time-validated** `const` preset), or `const`-unchecked from a trusted
/// literal/bake column with [`from_qualified`](Self::from_qualified).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NamedFeature<'a> {
    qualified: &'a str,
}

impl<'a> NamedFeature<'a> {
    /// Parse and validate a qualified `"name@hex8"` string, borrowing `s` — the validating
    /// constructor, no allocation, and `const` (so `const PRESET: NamedFeature =
    /// NamedFeature::parse("variance@b4a1c2d3").unwrap();` is checked at build time). Strict:
    /// a valid name, one `@`, exactly 8 **lowercase** hex digits. `None` on any deviation.
    #[must_use]
    pub const fn parse(s: &'a str) -> Option<Self> {
        let b = s.as_bytes();
        let n = b.len();
        let mut at = n;
        let mut i = 0;
        while i < n {
            if b[i] == b'@' {
                at = i;
                break;
            }
            i += 1;
        }
        if at == n || at == 0 {
            return None; // no '@', or empty name
        }
        let mut j = 0;
        while j < at {
            if !matches!(b[j], b'a'..=b'z' | b'0'..=b'9' | b'_') {
                return None;
            }
            j += 1;
        }
        if n - at - 1 != 8 {
            return None;
        }
        let mut k = at + 1;
        while k < n {
            if !matches!(b[k], b'0'..=b'9' | b'a'..=b'f') {
                return None;
            }
            k += 1;
        }
        Some(Self { qualified: s })
    }

    /// Wrap a qualified string WITHOUT validating — `const`, for trusted producers (a bake
    /// column you built with [`qualified_for`](Self::qualified_for)). Prefer [`parse`](Self::parse)
    /// for untrusted input.
    #[must_use]
    pub const fn from_qualified(qualified: &'a str) -> Self {
        Self { qualified }
    }

    /// The self-describing `"name@hex8"` form — the identity, and a parquet-safe column
    /// name (a drifted feature is a *different* column).
    #[must_use]
    pub const fn qualified_name(&self) -> &'a str {
        self.qualified
    }
    /// The feature name — the part before `@`, a zero-alloc split (cold path; reuse matches
    /// the whole qualified name). Falls back to the whole string if there is no `@`.
    #[must_use]
    pub fn name(&self) -> &'a str {
        match self.qualified.split_once('@') {
            Some((name, _)) => name,
            None => self.qualified,
        }
    }
    /// The folded ([`fold_hash`](Self::fold_hash)) 32-bit code-version hash — the hex after
    /// `@`, parsed on demand (`0` if ill-formed). Cold path / diagnostics.
    #[must_use]
    pub fn version_hash(&self) -> u32 {
        self.qualified
            .split_once('@')
            .and_then(|(_, hex)| u32::from_str_radix(hex, 16).ok())
            .unwrap_or(0)
    }

    /// Whether `name` is a valid feature name: a non-empty `[a-z0-9_]+`. Every name in this
    /// contract MUST satisfy this (parquet-column-safe, and the `name@hash` form stays
    /// unambiguous). `const`, so a preset can be checked at build time.
    #[must_use]
    pub const fn is_valid_name(name: &str) -> bool {
        let b = name.as_bytes();
        if b.is_empty() {
            return false;
        }
        let mut i = 0;
        while i < b.len() {
            if !matches!(b[i], b'a'..=b'z' | b'0'..=b'9' | b'_') {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Fold a full `zenanalyze::feature_version_hash()` (`u64`) into the 32-bit code-version
    /// the qualified name carries. Producers MUST route every hash through this so the fold
    /// is identical everywhere. `xor`-folding the halves keeps the full FNV distribution.
    #[must_use]
    pub const fn fold_hash(full: u64) -> u32 {
        ((full >> 32) ^ (full & 0xffff_ffff)) as u32
    }

    /// Allocate the qualified `"name@hex8"` form for a `(name, version_hash)` you only have
    /// at runtime — then wrap it with [`from_qualified`](Self::from_qualified). A producer's
    /// static vocabulary should carry the qualified form as a literal instead; this is the
    /// crate's only name allocation, and it's opt-in.
    #[must_use]
    pub fn qualified_for(name: &str, version_hash: u32) -> String {
        debug_assert!(
            Self::is_valid_name(name),
            "feature name must be [a-z0-9_]+: {name:?}"
        );
        let mut s = String::with_capacity(name.len() + 9);
        s.push_str(name);
        s.push('@');
        push_hex8(&mut s, version_hash);
        s
    }
}

impl fmt::Display for NamedFeature<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.qualified)
    }
}

impl<'a> TryFrom<&'a str> for NamedFeature<'a> {
    type Error = FormatError;
    fn try_from(s: &'a str) -> Result<Self, FormatError> {
        Self::parse(s).ok_or(FormatError::BadLine)
    }
}

// ──────────────────────────────── Value ────────────────────────────────────

/// A feature's native value — mirroring zenanalyze's output types, with the canonical
/// `f32` projection built in (the picker currency). `#[non_exhaustive]` so a future
/// structured type can land additively. Construct via `From` (`0.5f32`/`4096u32`/`true`
/// all `.into()`); the variants are public so a consumer that wants the native form can
/// match.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Value {
    /// A 32-bit float (most features).
    F32(f32),
    /// An unsigned 32-bit integer (counts, dims, bit depth, …).
    U32(u32),
    /// An unsigned 64-bit integer (domains beyond `u32::MAX`).
    U64(u64),
    /// A boolean flag (`alpha_present`, `is_grayscale`, …).
    Bool(bool),
}

impl Value {
    /// The **canonical** `f32` projection — the model-input currency. `Bool(false/true) →
    /// 0.0/1.0`, `U32(n) → n as f32` (exact for `n ≤ 2²⁴`), `U64(n) → n as f64 as f32`
    /// (exact for `n ≤ 2⁵³`). Identical to zenanalyze's `FeatureValue::to_f32`.
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        match self {
            Self::F32(x) => x,
            Self::U32(x) => x as f32,
            Self::U64(x) => x as f64 as f32,
            Self::Bool(false) => 0.0,
            Self::Bool(true) => 1.0,
        }
    }
}

impl From<f32> for Value {
    fn from(x: f32) -> Self {
        Self::F32(x)
    }
}
impl From<u32> for Value {
    fn from(x: u32) -> Self {
        Self::U32(x)
    }
}
impl From<u64> for Value {
    fn from(x: u64) -> Self {
        Self::U64(x)
    }
}
impl From<bool> for Value {
    fn from(x: bool) -> Self {
        Self::Bool(x)
    }
}

// ───────────────────────────── FeatureResult ───────────────────────────────

/// A feature's [`NamedFeature`] identity together with its computed [`Value`] — one cell of
/// a result. Private fields; build with [`new`](Self::new) (`value` is `impl Into<Value>`,
/// so `0.5f32` / `4096u32` / `true` all work). Read the native [`Value`] via
/// [`value`](Self::value), or the canonical `f32` (the picker currency) via
/// [`float`](Self::float).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureResult<'a> {
    feature: NamedFeature<'a>,
    value: Value,
}

impl<'a> FeatureResult<'a> {
    /// Pair an identity with a value (native type preserved).
    #[must_use]
    pub fn new(feature: NamedFeature<'a>, value: impl Into<Value>) -> Self {
        Self {
            feature,
            value: value.into(),
        }
    }
    /// The feature's identity.
    #[must_use]
    pub const fn feature(&self) -> NamedFeature<'a> {
        self.feature
    }
    /// The native [`Value`] — the feature's value with its exact type + precision.
    #[must_use]
    pub const fn value(&self) -> Value {
        self.value
    }
    /// The canonical `f32` projection of the value (the model-input currency) —
    /// [`Value::to_f32`].
    #[must_use]
    pub const fn float(&self) -> f32 {
        self.value.to_f32()
    }
    /// Shorthand for `self.feature().name()`.
    #[must_use]
    pub fn name(&self) -> &'a str {
        self.feature.name()
    }
}

// ───────────────────────────── OwnedFeatureResult ────────────────────────────────

/// The owned twin of [`FeatureResult`] — owns its qualified name and native [`Value`]; the
/// cell an [`OwnedOffer`] stores (e.g. a deserialized parquet row). Read it like a
/// `FeatureResult`, or lend a borrowed one via [`as_ref`](Self::as_ref).
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedFeatureResult {
    // `Box<str>` not `String`: the qualified name is write-once, so the spare capacity word a
    // `String` carries is dead weight — `Box<str>` is one word smaller per cell.
    qualified: Box<str>,
    value: Value,
}

impl OwnedFeatureResult {
    /// Own a `(qualified_name, value)` pair — from a parsed/deserialized row. (`From<FeatureResult>`
    /// owns a borrowed cell instead.)
    #[must_use]
    pub fn new(qualified_name: &str, value: impl Into<Value>) -> Self {
        Self {
            qualified: qualified_name.into(),
            value: value.into(),
        }
    }
    /// The qualified `"name@hex8"` identity.
    #[must_use]
    pub fn qualified_name(&self) -> &str {
        &self.qualified
    }
    /// The feature name (the part before `@`).
    #[must_use]
    pub fn name(&self) -> &str {
        NamedFeature::from_qualified(&self.qualified).name()
    }
    /// The 32-bit code-version hash (the hex after `@`).
    #[must_use]
    pub fn version_hash(&self) -> u32 {
        NamedFeature::from_qualified(&self.qualified).version_hash()
    }
    /// The native [`Value`] — the feature's value with its exact type + precision.
    #[must_use]
    pub const fn value(&self) -> Value {
        self.value
    }
    /// The canonical `f32` projection of the value (the model-input currency).
    #[must_use]
    pub const fn float(&self) -> f32 {
        self.value.to_f32()
    }
    /// Lend a borrowed [`FeatureResult`] — the owned→borrowed bridge.
    #[must_use]
    pub fn as_ref(&self) -> FeatureResult<'_> {
        FeatureResult::new(NamedFeature::from_qualified(&self.qualified), self.value)
    }
}

impl From<FeatureResult<'_>> for OwnedFeatureResult {
    fn from(r: FeatureResult<'_>) -> Self {
        Self {
            qualified: r.feature.qualified.into(),
            value: r.value,
        }
    }
}

// ───────────────────────────── Provenance ──────────────────────────────────

/// **Informational** record of the conditions a pass ran under — *not* a reuse gate (reuse
/// is by qualified name, and config/descriptor are constant within one pass). Mandatory
/// `analyzer_version`; optional config/descriptor via a builder. The config/descriptor feed
/// the serialization blend gate ([`Offer::schema_hash`]) and are there for debugging.
/// `#[non_exhaustive]` — the open-ended metadata bag; future optional context (a timestamp,
/// a source tag, more hashes) lands as additional `with_*` builders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Provenance<'a> {
    analyzer_version: &'a str,
    config_hash: u64,
    descriptor_hash: u64,
}

impl<'a> Provenance<'a> {
    /// Start from the mandatory `analyzer_version`; config/descriptor default to `0`.
    #[must_use]
    pub const fn new(analyzer_version: &'a str) -> Self {
        Self {
            analyzer_version,
            config_hash: 0,
            descriptor_hash: 0,
        }
    }
    /// Record the value-affecting analysis-config digest (`AnalysisQuery::config_hash()`).
    #[must_use]
    pub const fn with_config(self, config_hash: u64) -> Self {
        Self {
            analyzer_version: self.analyzer_version,
            config_hash,
            descriptor_hash: self.descriptor_hash,
        }
    }
    /// Record the value-affecting input-framing digest (primaries/transfer/alpha/diffuse-white).
    #[must_use]
    pub const fn with_descriptor(self, descriptor_hash: u64) -> Self {
        Self {
            analyzer_version: self.analyzer_version,
            config_hash: self.config_hash,
            descriptor_hash,
        }
    }
    /// The `zenanalyze` version that produced the values (informational).
    #[must_use]
    pub const fn analyzer_version(&self) -> &'a str {
        self.analyzer_version
    }
    /// The recorded analysis-config digest (`0` if unset).
    #[must_use]
    pub const fn config_hash(&self) -> u64 {
        self.config_hash
    }
    /// The recorded input-framing digest (`0` if unset).
    #[must_use]
    pub const fn descriptor_hash(&self) -> u64 {
        self.descriptor_hash
    }
}

// ──────────────────────────── Request / Select ─────────────────────────────

/// What a consumer wants extracted. Explicit wants ([`Select::Features`]) enable
/// per-feature reuse; [`Select::All`] is the build-relative "everything this provider can
/// produce". A "preset" is just a `const &[NamedFeature]` in the consumer, passed as
/// `Features`. `#[non_exhaustive]` keeps a future selector non-breaking.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Select<'a> {
    /// Everything the provider build can produce (resolved provider-side).
    All,
    /// An explicit list of qualified identities — the model's columns. **Version-pinned:**
    /// a code drift in any wanted feature is a miss, so a compiled model never silently
    /// eats a re-defined feature. This is the variant a picker/model MUST use.
    Features(&'a [NamedFeature<'a>]),
    /// Bare feature **names**, matched at *whatever* version the provider/offer carries —
    /// deliberately version-agnostic.
    ///
    /// For consumers whose use of a value is robust to a code-version drift: threshold
    /// heuristics and content classifiers, diagnostics, and bulk column export. It lets such
    /// a consumer name features **without** naming a `zenanalyze` version — the reason it
    /// exists, since [`Features`](Self::Features) would otherwise force a consumer to
    /// hard-code version hashes it cannot know across builds.
    ///
    /// **Never feed a compiled model from `Names`.** A model's coefficients were fit against
    /// one code version of each column; matching by bare name would silently substitute a
    /// re-defined feature and corrupt the prediction with no error. Use
    /// [`Features`](Self::Features) there — that is exactly the miss `Names` gives up.
    Names(&'a [&'a str]),
}

/// A consumer's ask — its [`Select`]. Build with [`Request::new`]; private field (read via
/// [`select`](Self::select)) so a future request-options bag (`with_*`) lands additively.
/// No `config_hash` — the config the consumer needs is already constant within a pass.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    select: Select<'a>,
}

impl<'a> Request<'a> {
    /// Build a request.
    #[must_use]
    pub const fn new(select: Select<'a>) -> Self {
        Self { select }
    }
    /// What to extract / reuse.
    #[must_use]
    pub const fn select(&self) -> Select<'a> {
        self.select
    }
}

// ──────────────────────────────── Offer ────────────────────────────────────

/// A self-describing feature result — the SAME thing in memory and on disk. Carries every
/// [`FeatureResult`] plus the informational [`Provenance`]. Private fields (read via the
/// accessors); build with [`Offer::new`].
#[derive(Clone, Copy, Debug)]
pub struct Offer<'a> {
    features: &'a [FeatureResult<'a>],
    provenance: Provenance<'a>,
}

impl<'a> Offer<'a> {
    /// Build an offer from a completed extraction and its [`Provenance`].
    #[must_use]
    pub const fn new(features: &'a [FeatureResult<'a>], provenance: Provenance<'a>) -> Self {
        Self {
            features,
            provenance,
        }
    }
    /// The computed features (identity + value).
    #[must_use]
    pub const fn features(&self) -> &'a [FeatureResult<'a>] {
        self.features
    }
    /// The conditions these values were produced under (informational).
    #[must_use]
    pub const fn provenance(&self) -> Provenance<'a> {
        self.provenance
    }
    /// The result for `name` (by **bare name**, any version), or `None` if absent. By-name is
    /// what lets a consumer classify a reuse miss without a re-run: `None` ⇒ missing; `Some`
    /// whose [`version_hash`](NamedFeature::version_hash) ≠ the wanted one ⇒ a code drift.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&FeatureResult<'a>> {
        self.features.iter().find(|f| f.feature.name() == name)
    }
    /// Whether this offer can fully satisfy `req` — the codec's "do I need to re-run?" check,
    /// without materializing values (`reuse_for(req).is_some()` minus the alloc). When it
    /// can't and a re-run isn't possible, loop [`get`](Self::get) per want for the *why*.
    #[must_use]
    pub fn satisfies(&self, req: &Request<'_>) -> bool {
        satisfies_impl(self.features, req)
    }
    /// Reuse this offer for `req`, all-or-nothing, in request order, as canonical `f32` —
    /// `Some(values)` iff [`satisfies`](Self::satisfies); else `None` (run own pass).
    #[must_use]
    pub fn reuse_for(&self, req: &Request<'_>) -> Option<Vec<f32>> {
        reuse_impl(self.features, req)
    }
    /// A single `u64` over the SET of qualified names plus the provenance `config_hash` +
    /// `descriptor_hash` — order-independent. **The file-blend gate:** two offers/files with
    /// equal `schema_hash` carry identical columns under identical conditions.
    #[must_use]
    pub fn schema_hash(&self) -> u64 {
        schema_hash_of(
            self.features.iter().map(|f| f.feature.qualified),
            self.provenance.config_hash,
            self.provenance.descriptor_hash,
        )
    }
    /// Serialize to the dependency-free `zenanalyze-features/1` **text** block — one
    /// `name@hex8=value` line per feature (value carrying its native type,
    /// `true`/`4096_u32`/`0.5`, so precision survives), round-tripping through
    /// [`OwnedOffer::parse`]. For bulk data prefer columnar parquet/TSV, reading the pieces
    /// directly ([`features`](Self::features) + [`provenance`](Self::provenance) +
    /// [`schema_hash`](Self::schema_hash) → columns + metadata); this text form is for debug
    /// dumps, tests, and compact self-describing stamps.
    #[must_use]
    pub fn to_block(&self) -> String {
        let p = self.provenance;
        let mut s = String::new();
        s.push_str(MAGIC);
        s.push('\n');
        s.push_str(&format!("analyzer_version={}\n", p.analyzer_version));
        s.push_str(&format!("config_hash={}\n", p.config_hash));
        s.push_str(&format!("descriptor_hash={}\n", p.descriptor_hash));
        s.push_str(&format!("schema_hash={}\n", self.schema_hash()));
        s.push_str("[features]\n");
        for fr in self.features {
            s.push_str(fr.feature.qualified);
            s.push('=');
            push_value(&mut s, fr.value);
            s.push('\n');
        }
        s
    }
}

// ──────────────────────────────── Catalog ──────────────────────────────────

/// What a provider build can produce — its available qualified identities. Compile-feature
/// availability surfaces here: a feature gated out of the build is simply absent. Private
/// field (read via [`available`](Self::available)); build with [`Catalog::new`].
#[derive(Clone, Copy, Debug)]
pub struct Catalog<'a> {
    available: &'a [NamedFeature<'a>],
}

impl<'a> Catalog<'a> {
    /// Build a catalog.
    #[must_use]
    pub const fn new(available: &'a [NamedFeature<'a>]) -> Self {
        Self { available }
    }
    /// The identities this build can extract.
    #[must_use]
    pub const fn available(&self) -> &'a [NamedFeature<'a>] {
        self.available
    }
    /// Whether the build can produce `want` at exactly its version — an exact qualified-name
    /// match (⇒ reusable).
    #[must_use]
    pub fn offers(&self, want: &NamedFeature<'_>) -> bool {
        self.available.iter().any(|a| a.qualified == want.qualified)
    }
    /// Whether the build has a feature by `name` (possibly at a different version).
    #[must_use]
    pub fn has_name(&self, name: &str) -> bool {
        self.available.iter().any(|a| a.name() == name)
    }
    /// The wanted names this build can't produce at the wanted version — what a consumer must
    /// own-pass (or, if absent locally, a build error).
    #[must_use]
    pub fn unmet<'w>(&self, wants: &'w [NamedFeature<'_>]) -> Vec<&'w str> {
        wants
            .iter()
            .filter(|w| !self.offers(w))
            .map(|w| w.name())
            .collect()
    }
    /// The distinct feature names one pass over `requests` should extract — the "unionize"
    /// step. Resolves [`Select::All`] against this catalog; explicit [`Select::Features`] /
    /// [`Select::Names`] use their own names. First-seen order, deduplicated.
    #[must_use]
    pub fn union(&self, requests: &[Request<'a>]) -> Vec<&'a str> {
        union_impl(self.available, requests)
    }
}

// ───────────────────────────── OwnedCatalog ────────────────────────────────

/// The owned twin of [`Catalog`] — owns its qualified names, so a provider whose vocabulary
/// is only known at runtime (qualified names built by
/// [`NamedFeature::qualified_for`](NamedFeature::qualified_for)) can still publish one.
/// [`FeatureProvider::catalog`] returns this. Same queries as `Catalog`; lend borrowed
/// identities with [`available`](Self::available).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OwnedCatalog {
    // `Box<str>` for the same reason as `OwnedFeatureResult`: write-once, so a `String`'s
    // spare-capacity word is dead weight.
    available: Vec<Box<str>>,
}

impl OwnedCatalog {
    /// Collect a build's qualified `"name@hex8"` identities.
    #[must_use]
    pub fn new(qualified_names: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        Self {
            available: qualified_names
                .into_iter()
                .map(|q| q.as_ref().into())
                .collect(),
        }
    }
    /// The identities this build can extract, lent as borrowed [`NamedFeature`]s (the
    /// owned→borrowed bridge; zero-alloc).
    pub fn available(&self) -> impl Iterator<Item = NamedFeature<'_>> + '_ {
        self.available
            .iter()
            .map(|q| NamedFeature::from_qualified(q))
    }
    /// How many identities the build offers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.available.len()
    }
    /// Whether the build offers nothing (a provider that cannot extract).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.available.is_empty()
    }
    /// Whether the build can produce `want` at exactly its version — exactly [`Catalog::offers`].
    #[must_use]
    pub fn offers(&self, want: &NamedFeature<'_>) -> bool {
        self.available.iter().any(|a| &**a == want.qualified)
    }
    /// Whether the build has a feature by `name` (possibly at a different version).
    #[must_use]
    pub fn has_name(&self, name: &str) -> bool {
        self.available().any(|a| a.name() == name)
    }
    /// The wanted names this build can't produce at the wanted version — exactly
    /// [`Catalog::unmet`].
    #[must_use]
    pub fn unmet<'w>(&self, wants: &'w [NamedFeature<'_>]) -> Vec<&'w str> {
        wants
            .iter()
            .filter(|w| !self.offers(w))
            .map(|w| w.name())
            .collect()
    }
    /// The distinct names one pass over `requests` should extract — exactly [`Catalog::union`],
    /// resolving [`Select::All`] against this catalog.
    #[must_use]
    pub fn union<'s>(&'s self, requests: &[Request<'s>]) -> Vec<&'s str> {
        let available: Vec<NamedFeature<'s>> = self.available().collect();
        union_impl(&available, requests)
    }
}

// ──────────────────────── FeatureProvider / ProviderError ──────────────────

/// Why a [`FeatureProvider`] could not produce an offer. `#[non_exhaustive]`. Implements
/// `core::error::Error`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProviderError {
    /// The pixel buffer's length doesn't match `width * height * channels`, or a dimension
    /// was zero / out of range.
    BadInput,
    /// The provider build cannot produce something the [`Request`] asked for — check
    /// [`OwnedCatalog::unmet`] for which.
    Unavailable,
    /// An allocation failed.
    OutOfMemory,
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BadInput => "pixel buffer length or dimensions are invalid",
            Self::Unavailable => "this provider build cannot produce a requested feature",
            Self::OutOfMemory => "allocation failed",
        })
    }
}

impl core::error::Error for ProviderError {}

/// **The intermediary**: extraction reached through the contract instead of through a
/// `zenanalyze` type.
///
/// A codec that needs feature values but has no [`Offer`] to reuse would otherwise have to
/// depend on a concrete `zenanalyze` — and that is precisely the version pin that stops two
/// codecs from linking together. Taking a `&dyn FeatureProvider` instead keeps the codec's
/// only zenanalyze-family dependency this crate: the **host** picks the `zenanalyze` version,
/// implements this trait over it (`zenanalyze`'s own impl is behind its `api` feature), and
/// injects it. Two codecs built against different `zenanalyze` versions can then coexist,
/// each speaking the one contract.
///
/// Object-safe by construction — `&dyn FeatureProvider` is the intended form.
///
/// Implementors: honor the [`Request`] exactly. [`Select::Features`] wants those identities at
/// **that** code version, [`Select::Names`] at any version, [`Select::All`] is everything in
/// [`catalog`](Self::catalog). Returning an offer that silently drops a want is a contract
/// violation — either produce it or return [`ProviderError::Unavailable`] (a consumer's
/// [`satisfies`](Offer::satisfies) check is a reuse decision, not a correctness backstop).
pub trait FeatureProvider {
    /// The provider's `zenanalyze` version string — informational, and what it stamps on
    /// [`Provenance::analyzer_version`].
    fn analyzer_version(&self) -> &str;

    /// What this provider build can produce. Cold path; building it may allocate.
    fn catalog(&self) -> OwnedCatalog;

    /// Extract `request` from a tightly-packed 8-bit **sRGB** RGB buffer —
    /// `rgb.len() == width * height * 3`, no row padding — and bundle it as an
    /// [`OwnedOffer`] stamped with this provider's [`Provenance`].
    fn extract_rgb8(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        request: &Request<'_>,
    ) -> Result<OwnedOffer, ProviderError>;
}

// ──────────────────────────── OwnedOffer / parse ───────────────────────────

const MAGIC: &str = "zenanalyze-features/1";

/// Why a `zenanalyze-features/1` block could not be parsed. `#[non_exhaustive]`. Implements
/// `core::error::Error`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// The first line was not the recognized `zenanalyze-features/<v>` magic.
    UnknownFormat,
    /// A required header (`analyzer_version`/`config_hash`/`descriptor_hash`) absent.
    MissingHeader,
    /// A `key=value` / `name@hash=value` line was malformed, or a number/name didn't parse.
    BadLine,
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnknownFormat => "unrecognized zenanalyze-features magic",
            Self::MissingHeader => "a required header is missing",
            Self::BadLine => "a malformed line",
        })
    }
}

impl core::error::Error for FormatError {}

/// The owned twin of [`Offer`] — the on-disk side, a `Vec` of [`OwnedFeatureResult`] cells.
/// Build it from already-deserialized parts with [`OwnedOffer::new`] (the parquet / TSV path:
/// value columns → cells, metadata → [`Provenance`]) or from the text block with
/// [`OwnedOffer::parse`]. It carries the SAME negotiation surface as `Offer`
/// ([`satisfies`](Self::satisfies) / [`reuse_for`](Self::reuse_for) / [`get`](Self::get)) over its
/// owned cells, so a deserialized offer negotiates directly. [`features`](Self::features) lends the
/// cells zero-cost; map [`OwnedFeatureResult::as_ref`] over them + [`provenance`](Self::provenance)
/// to rebuild a borrowed [`Offer`].
#[derive(Clone, Debug)]
pub struct OwnedOffer {
    analyzer_version: String,
    config_hash: u64,
    descriptor_hash: u64,
    features: Vec<OwnedFeatureResult>,
}

impl OwnedOffer {
    /// Build an owned offer from already-deserialized parts — the primary path for a parquet /
    /// TSV reader (value columns → [`OwnedFeatureResult`]s, file metadata → [`Provenance`]).
    /// Mirrors [`Offer::new`]; [`parse`](Self::parse) is the dependency-free text alternative.
    #[must_use]
    pub fn new(features: Vec<OwnedFeatureResult>, provenance: Provenance<'_>) -> Self {
        Self {
            analyzer_version: provenance.analyzer_version.to_string(),
            config_hash: provenance.config_hash,
            descriptor_hash: provenance.descriptor_hash,
            features,
        }
    }

    /// Parse a `zenanalyze-features/1` block. Strict on the magic, the three required
    /// headers, a valid qualified name per feature key, and a parseable typed value;
    /// forward-compatible on unknown headers (ignored, incl. `schema_hash`, recomputed).
    pub fn parse(text: &str) -> Result<Self, FormatError> {
        let mut lines = text.lines();
        if lines.next().map(str::trim) != Some(MAGIC) {
            return Err(FormatError::UnknownFormat);
        }
        let (mut av, mut cfg, mut desc) = (None, None, None);
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
            let (key, val) = line.split_once('=').ok_or(FormatError::BadLine)?;
            let (key, val) = (key.trim(), val.trim());
            if in_features {
                if NamedFeature::parse(key).is_none() {
                    return Err(FormatError::BadLine);
                }
                let value = parse_value(val).ok_or(FormatError::BadLine)?;
                features.push(OwnedFeatureResult::new(key, value));
            } else {
                match key {
                    "analyzer_version" => av = Some(val.to_string()),
                    "config_hash" => cfg = Some(val.parse().map_err(|_| FormatError::BadLine)?),
                    "descriptor_hash" => {
                        desc = Some(val.parse().map_err(|_| FormatError::BadLine)?)
                    }
                    _ => {}
                }
            }
        }
        Ok(Self {
            analyzer_version: av.ok_or(FormatError::MissingHeader)?,
            config_hash: cfg.ok_or(FormatError::MissingHeader)?,
            descriptor_hash: desc.ok_or(FormatError::MissingHeader)?,
            features,
        })
    }

    fn results(&self) -> Vec<FeatureResult<'_>> {
        self.features
            .iter()
            .map(OwnedFeatureResult::as_ref)
            .collect()
    }

    /// The conditions these values were produced under (informational), borrowing this offer.
    #[must_use]
    pub fn provenance(&self) -> Provenance<'_> {
        Provenance::new(&self.analyzer_version)
            .with_config(self.config_hash)
            .with_descriptor(self.descriptor_hash)
    }
    /// The owned feature cells (zero-cost borrow). Map [`OwnedFeatureResult::as_ref`] over them +
    /// [`provenance`](Self::provenance) to rebuild a borrowed [`Offer`].
    #[must_use]
    pub fn features(&self) -> &[OwnedFeatureResult] {
        &self.features
    }
    /// The owned cell for `name` (by bare name), or `None` if absent.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&OwnedFeatureResult> {
        self.features.iter().find(|f| f.name() == name)
    }
    /// Whether this offer can fully satisfy `req` — exactly [`Offer::satisfies`].
    #[must_use]
    pub fn satisfies(&self, req: &Request<'_>) -> bool {
        satisfies_impl(&self.results(), req)
    }
    /// Reuse for `req`, exactly [`Offer::reuse_for`].
    #[must_use]
    pub fn reuse_for(&self, req: &Request<'_>) -> Option<Vec<f32>> {
        reuse_impl(&self.results(), req)
    }
    /// The same blend gate as [`Offer::schema_hash`], recomputed.
    #[must_use]
    pub fn schema_hash(&self) -> u64 {
        schema_hash_of(
            self.features.iter().map(OwnedFeatureResult::qualified_name),
            self.config_hash,
            self.descriptor_hash,
        )
    }
}

// ───────────────────────────── shared impls ────────────────────────────────

fn satisfies_impl(features: &[FeatureResult<'_>], req: &Request<'_>) -> bool {
    match req.select {
        Select::All => true,
        Select::Features(wants) => wants
            .iter()
            .all(|w| features.iter().any(|f| f.feature.qualified == w.qualified)),
        Select::Names(wants) => wants
            .iter()
            .all(|w| features.iter().any(|f| f.feature.name() == *w)),
    }
}

fn reuse_impl(features: &[FeatureResult<'_>], req: &Request<'_>) -> Option<Vec<f32>> {
    match req.select {
        Select::All => Some(features.iter().map(|f| f.float()).collect()),
        Select::Features(wants) => wants
            .iter()
            .map(|w| {
                features
                    .iter()
                    .find(|f| f.feature.qualified == w.qualified)
                    .map(|f| f.float())
            })
            .collect(),
        Select::Names(wants) => wants
            .iter()
            .map(|w| {
                features
                    .iter()
                    .find(|f| f.feature.name() == *w)
                    .map(|f| f.float())
            })
            .collect(),
    }
}

/// The distinct bare names `requests` asks for, resolving [`Select::All`] against
/// `available`. Shared by [`Catalog::union`] and [`OwnedCatalog::union`].
fn union_impl<'s>(available: &[NamedFeature<'s>], requests: &[Request<'s>]) -> Vec<&'s str> {
    let mut out: Vec<&'s str> = Vec::new();
    let mut push = |name: &'s str| {
        if !out.contains(&name) {
            out.push(name);
        }
    };
    for r in requests {
        match r.select {
            Select::All => {
                for nf in available {
                    push(nf.name());
                }
            }
            Select::Features(w) => {
                for nf in w {
                    push(nf.name());
                }
            }
            Select::Names(w) => {
                for name in w {
                    push(name);
                }
            }
        }
    }
    out
}

// ───────────────────────────── internal helpers ────────────────────────────

fn push_value(s: &mut String, v: Value) {
    match v {
        Value::F32(x) => s.push_str(&format!("{x}")),
        Value::U32(x) => s.push_str(&format!("{x}_u32")),
        Value::U64(x) => s.push_str(&format!("{x}_u64")),
        Value::Bool(b) => s.push_str(if b { "true" } else { "false" }),
    }
}

fn parse_value(s: &str) -> Option<Value> {
    match s {
        "true" => return Some(Value::Bool(true)),
        "false" => return Some(Value::Bool(false)),
        _ => {}
    }
    if let Some(n) = s.strip_suffix("_u32") {
        return n.parse().ok().map(Value::U32);
    }
    if let Some(n) = s.strip_suffix("_u64") {
        return n.parse().ok().map(Value::U64);
    }
    s.parse().ok().map(Value::F32)
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(FNV_PRIME);
    }
    h
}

/// Order-independent digest of a set of qualified names + the two recorded axes.
fn schema_hash_of<'i>(
    qualified: impl Iterator<Item = &'i str>,
    config: u64,
    descriptor: u64,
) -> u64 {
    let mut v: Vec<&str> = qualified.collect();
    v.sort_unstable();
    let mut h = FNV_OFFSET;
    for q in &v {
        h = fnv(h, q.as_bytes());
        h = fnv(h, &[0]);
    }
    h = fnv(h, &config.to_le_bytes());
    fnv(h, &descriptor.to_le_bytes())
}

fn push_hex8(s: &mut String, v: u32) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for i in (0..8).rev() {
        s.push(HEX[((v >> (i * 4)) & 0xf) as usize] as char);
    }
}

#[cfg(test)]
mod tests;
