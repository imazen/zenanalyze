//! Codec-family meta-picker — given image features, a quality
//! target, and the caller's allowed-family mask, choose a codec
//! family. Per-codec pickers (separate ZNPR v2 bakes shipped by the
//! codec crate) then resolve the family into a concrete encoder
//! config.
//!
//! ## Where it sits
//!
//! ```text
//!     features (zenanalyze) + target_zq + caller constraints
//!                            │
//!                            ▼
//!                ┌──────────────────────┐
//!                │ zenpicker::MetaPicker│   one ZNPR v2 model;
//!                │  argmin over family  │   N_outputs = N families
//!                └──────────┬───────────┘
//!                           │ chosen family
//!                           ▼
//!                ┌──────────────────────┐
//!                │ Per-codec picker     │   one .bin per family,
//!                │  (zenpredict model)  │   shipped from the codec
//!                │  → cell + scalars    │   crate
//!                └──────────┬───────────┘
//!                           ▼
//!                  concrete EncoderConfig
//! ```
//!
//! The meta-picker emits a [`CodecFamily`]; it does **not** know how
//! to resolve a family into a concrete encoder config. That's the
//! job of the family's per-codec picker (a separate ZNPR v2 bake
//! shipped by the codec crate, also loaded via [`zenpredict`]).
//!
//! ## Wire format
//!
//! Internally a [`MetaPicker`] is just a [`zenpredict::Predictor`]
//! whose `n_outputs` equals [`CodecFamily::COUNT`]. The output index
//! is the family enum's discriminant; bake-time and runtime must
//! agree on the order via the model's metadata
//! ([`FAMILY_ORDER_KEY`] = `zenpicker.family_order`, UTF-8,
//! comma-separated lower-case labels — same order as
//! [`CodecFamily::ALL`]).
//!
//! ## Crate boundary
//!
//! - [`zenpredict`] — the runtime this crate composes on. Owns the
//!   ZNPR v2 binary format, the parser, the forward pass, the
//!   masked-argmin math, the metadata blob, and the `Predictor`.
//!   `zenpicker` adds: family enum + family-order validation +
//!   `AllowedFamilies` mask sugar.
//! - [`zentrain`](https://github.com/imazen/zenanalyze/tree/main/zentrain)
//!   — Python training pipeline that produces the `.bin` artifact a
//!   meta-picker (or a per-codec picker) loads. Train with
//!   `cells = families` and `output_layout = bytes_log` only
//!   (purely categorical, no scalar heads).
//! - [`zenanalyze`](https://crates.io/crates/zenanalyze) — feature
//!   extractor that produces the input vector both this meta-picker
//!   and the per-codec pickers consume.
//!
//! ## Status
//!
//! v0.1 establishes the crate boundary and the API shape. Baking an
//! actual cross-codec meta-picker model is downstream work — once a
//! labelled training set exists where each row maps `(image
//! features, target_zq) → best family`, run zentrain's
//! `train_hybrid.py` with `cells = families` and `output_layout` of
//! `bytes_log` only.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};

use zenpredict::{AllowedMask, ArgminOffsets, Model, PredictError, Predictor, ScoreTransform};

mod route;
pub use route::{
    LOSSLESS_PREFERENCE, LOSSLESS_QUALITY, LOSSY_PREFERENCE, QualityTarget, RouteDecision,
};
#[cfg(feature = "api")]
pub use route::{content_capability, family_rule};

// ── Shipped default cross-codec routers (baked 2026-06-30, ZNPR) ──────────
// gate + lossless are i8 family-score MLPs; the lossy router is the f32 6-pairwise-discriminant
// linear model (102→6 margins) combined by `route::pairwise_round_robin`.
// The ZNPR loader needs 16-aligned bytes; wrap each baked blob in an over-aligned
// struct (same pattern the per-codec pickers use). `default_routers()` loads them.
#[cfg(feature = "std")]
#[repr(C, align(16))]
struct AlignedModel<const N: usize>([u8; N]);
#[cfg(feature = "std")]
const ROUTER_LOSSY: &[u8] = &AlignedModel(*include_bytes!(
    "../benchmarks/zenpicker_router_lossy_v0.1.bin"
))
.0;
#[cfg(feature = "std")]
const ROUTER_LOSSLESS: &[u8] = &AlignedModel(*include_bytes!(
    "../benchmarks/zenpicker_router_lossless_v0.1.bin"
))
.0;
#[cfg(feature = "std")]
const ROUTER_GATE: &[u8] = &AlignedModel(*include_bytes!(
    "../benchmarks/zenpicker_router_gate_v0.1.bin"
))
.0;

/// Codec families the meta-picker can choose between.
///
/// **Important — order matters.** The discriminants here must match
/// the order in the baked meta-picker model's output vector. Bakes
/// declare the order via the model metadata ([`FAMILY_ORDER_KEY`]).
/// Runtime checks this at load via [`MetaPicker::validate_family_order`].
///
/// Adding a new family is a breaking change for any baked model that
/// existed before — the schema_hash will mismatch and the runtime
/// will refuse to load the old model. Bake a fresh meta-picker that
/// includes the new family before deploying the codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum CodecFamily {
    Jpeg = 0,
    Webp = 1,
    Jxl = 2,
    Avif = 3,
    Png = 4,
    Gif = 5,
}

impl CodecFamily {
    /// Number of variants currently defined. Used to size masks /
    /// allocate output buffers. Bump this when adding a variant.
    pub const COUNT: usize = 6;

    /// All variants in declared order — same order the bake's
    /// `output_layout` must use.
    pub const ALL: [CodecFamily; Self::COUNT] = [
        Self::Jpeg,
        Self::Webp,
        Self::Jxl,
        Self::Avif,
        Self::Png,
        Self::Gif,
    ];

    /// Discriminant as `usize` for indexing into mask / output arrays.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Stable string label.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
            Self::Jxl => "jxl",
            Self::Avif => "avif",
            Self::Png => "png",
            Self::Gif => "gif",
        }
    }
}

/// Caller-supplied filter over which families are acceptable for a
/// given encode. Wraps a fixed-size `[bool; CodecFamily::COUNT]` so
/// the runtime can build a [`zenpredict::AllowedMask`] without
/// allocating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllowedFamilies {
    flags: [bool; CodecFamily::COUNT],
}

impl AllowedFamilies {
    pub const fn none() -> Self {
        Self {
            flags: [false; CodecFamily::COUNT],
        }
    }

    pub const fn all() -> Self {
        Self {
            flags: [true; CodecFamily::COUNT],
        }
    }

    /// Build an `AllowedFamilies` from an iterator over allowed
    /// families (everything else denied). Named `from` rather than
    /// `from_iter` to keep clippy's `should_implement_trait` lint
    /// happy without claiming the full `FromIterator` contract
    /// (which would force `Self::from_iter(empty()) == none()`,
    /// matching what we do — but spelling out the trait pulls in
    /// blanket impls we don't need).
    pub fn from_allowed<I: IntoIterator<Item = CodecFamily>>(iter: I) -> Self {
        let mut me = Self::none();
        for fam in iter {
            me.flags[fam.index()] = true;
        }
        me
    }

    pub fn allow(mut self, fam: CodecFamily) -> Self {
        self.flags[fam.index()] = true;
        self
    }

    pub fn deny(mut self, fam: CodecFamily) -> Self {
        self.flags[fam.index()] = false;
        self
    }

    pub fn is_allowed(self, fam: CodecFamily) -> bool {
        self.flags[fam.index()]
    }

    pub const fn as_slice(&self) -> &[bool] {
        &self.flags
    }

    pub fn any(self) -> bool {
        self.flags.iter().any(|f| *f)
    }

    /// Intersection — families allowed by BOTH masks. Folds the caller allowlist with
    /// [`content_capability`] (and the branch set) in `route`.
    pub fn intersect(self, other: Self) -> Self {
        let mut flags = [false; CodecFamily::COUNT];
        let mut i = 0;
        while i < CodecFamily::COUNT {
            flags[i] = self.flags[i] && other.flags[i];
            i += 1;
        }
        Self { flags }
    }

    /// The families the LOSSY router chooses among (JPEG / WebP / JXL / AVIF).
    pub const LOSSY: Self = Self {
        flags: [true, true, true, true, false, false],
    };
    /// The families the LOSSLESS router chooses among (WebP / JXL / PNG).
    pub const LOSSLESS: Self = Self {
        flags: [false, true, true, false, true, false],
    };

    /// Filter to families whose estimated encode cost fits a real-time latency ceiling.
    /// Real-time [`EncodeMode`]s drop families slower than `latency_ms`; queued modes
    /// keep every already-allowed family (they can wait). `per_family_est_ms[fam.index()]`
    /// is the codec's own encode-time estimate for this image — codecs have cost models,
    /// so the caller never guesses. `latency_ms = None` applies no time gate.
    ///
    /// [`EncodeMode`]: zenpredict::EncodeMode
    pub fn viable(
        self,
        mode: zenpredict::EncodeMode,
        latency_ms: Option<u32>,
        per_family_est_ms: &[u32; CodecFamily::COUNT],
    ) -> Self {
        if !mode.is_realtime() {
            return self;
        }
        let mut out = Self::none();
        for fam in CodecFamily::ALL {
            if self.is_allowed(fam)
                && latency_ms.is_none_or(|c| per_family_est_ms[fam.index()] <= c)
            {
                out = out.allow(fam);
            }
        }
        out
    }
}

/// Metadata key the bake declares to assert family-order agreement
/// between trainer and runtime.
pub const FAMILY_ORDER_KEY: &str = "zenpicker.family_order";

/// Expected value of [`FAMILY_ORDER_KEY`] for the current
/// [`CodecFamily::ALL`] layout.
pub const ALL_LABELS_CSV: &str = "jpeg,webp,jxl,avif,png,gif";

/// Metadata key a **pairwise** lossy router carries (the shipped default lossy router): its value
/// is the CSV of the 6 codec pairs in output-neuron order. A model with this key emits per-pair
/// margins, **not** per-family scores — so [`MetaPicker::pick`] (a raw argmin) refuses it; use
/// [`route`](MetaPicker::route), which applies the round-robin (`route::pairwise_round_robin`).
pub const LOSSY_PAIRWISE_KEY: &str = "zenpicker.lossy_pairwise";

/// One thin meta-picker.
///
/// Owns a [`zenpredict::Predictor`] (mutable scratch for one forward
/// pass at a time). Construction is cheap; reuse a single instance
/// across many encode requests.
pub struct MetaPicker<'b> {
    predictor: Predictor<'b>,
    /// The model's feature columns parsed once into `zenanalyze_api::NamedFeature`
    /// identities (qualified `name@hex8`), so [`feature_request`](Self::feature_request)
    /// can lend them inside a `Select::Features` without re-parsing per call. `Some` only
    /// when **every** column is a valid qualified name; `None` for a pre-`name@hash` bake
    /// (or no columns), so such a model can't reuse and always runs its own pass — never a
    /// wrong-length vector from a vacuously-satisfied empty request.
    #[cfg(feature = "api")]
    wants: Option<alloc::vec::Vec<zenanalyze_api::NamedFeature<'b>>>,
    /// The auto-gate (lossy|lossless, 2 outputs) and the lossless family model, set by
    /// [`with_router`](Self::with_router). `None` for a bare single-model picker — only
    /// [`pick`](Self::pick) works then; [`route`](Self::route) needs the full set.
    gate: Option<Predictor<'b>>,
    lossless: Option<Predictor<'b>>,
    #[cfg(feature = "api")]
    gate_wants: Option<alloc::vec::Vec<zenanalyze_api::NamedFeature<'b>>>,
    #[cfg(feature = "api")]
    lossless_wants: Option<alloc::vec::Vec<zenanalyze_api::NamedFeature<'b>>>,
}

/// Parse a model's feature columns into qualified `name@hex8` identities, or `None` if any
/// column isn't qualified (a pre-`name@hash` bake) — such a model can't reuse a shared offer.
#[cfg(feature = "api")]
fn parse_wants(model: &Model) -> Option<alloc::vec::Vec<zenanalyze_api::NamedFeature<'_>>> {
    let cols: alloc::vec::Vec<&str> = model.feature_columns().collect();
    let parsed: alloc::vec::Vec<_> = cols
        .iter()
        .copied()
        .filter_map(zenanalyze_api::NamedFeature::parse)
        .collect();
    (!cols.is_empty() && parsed.len() == cols.len()).then_some(parsed)
}

/// Materialize a model's input vector from a shared offer (reuse its feature columns), append
/// an optional scalar routing input (the target quality), and run the forward pass.
/// `Ok(None)` when the offer can't satisfy the model's columns — the caller runs its own pass.
#[cfg(feature = "api")]
fn score(
    pred: &mut Predictor<'_>,
    wants: Option<&[zenanalyze_api::NamedFeature<'_>]>,
    offer: &zenanalyze_api::Offer<'_>,
    extra: Option<f32>,
) -> Result<Option<alloc::vec::Vec<f32>>, MetaPickerError> {
    let Some(w) = wants else { return Ok(None) };
    let req = zenanalyze_api::Request::new(zenanalyze_api::Select::Features(w));
    let Some(mut x) = offer.reuse_for(&req) else {
        return Ok(None);
    };
    if let Some(e) = extra {
        x.push(e);
    }
    let out = pred.predict(&x).map_err(MetaPickerError::Predict)?;
    Ok(Some(out.to_vec()))
}

impl<'b> MetaPicker<'b> {
    /// Wrap a parsed [`zenpredict::Model`]. Caller is expected to
    /// have validated the schema hash via
    /// [`zenpredict::Model::from_bytes_with_schema`] or by reading
    /// the model's metadata.
    ///
    /// Call [`MetaPicker::validate_family_order`] right after
    /// construction to confirm bake-time and runtime agree on the
    /// family enum layout.
    ///
    /// Takes `&'b Model` — the Model is the long-lived parsed bake
    /// (typically inside a `static OnceLock<Model>`); the borrow
    /// lifetime `'b` flows through into `Predictor<'b>` and then
    /// `MetaPicker<'b>`. zenpredict 0.2.0+ made Model own its
    /// reference-data rather than carry a lifetime parameter.
    pub fn new(model: &'b Model) -> Self {
        Self {
            #[cfg(feature = "api")]
            wants: parse_wants(model),
            predictor: Predictor::new(model),
            gate: None,
            lossless: None,
            #[cfg(feature = "api")]
            gate_wants: None,
            #[cfg(feature = "api")]
            lossless_wants: None,
        }
    }

    /// Promote a single-model picker into the full quality-aware cross-codec router: this
    /// picker's own model becomes the **lossy** family router, and `gate` (a 2-output
    /// lossy|lossless auto-gate) + `lossless` (the lossless family router) are added. After
    /// this, [`route`](Self::route) is available. Each model is a separate ZNPR bake; the two
    /// family routers score [`CodecFamily::COUNT`] outputs (validate each with
    /// [`validate_family_order`](Self::validate_family_order)); the gate scores 2.
    pub fn with_router(mut self, gate: &'b Model, lossless: &'b Model) -> Self {
        self.gate = Some(Predictor::new(gate));
        self.lossless = Some(Predictor::new(lossless));
        #[cfg(feature = "api")]
        {
            self.gate_wants = parse_wants(gate);
            self.lossless_wants = parse_wants(lossless);
        }
        self
    }

    /// Build a [`zenanalyze_api::Request`] for this picker's model — its feature columns as
    /// qualified `name@hex8` identities (each carrying the per-feature **code** version).
    ///
    /// A caller negotiates a shared [`Offer`](zenanalyze_api::Offer) with
    /// `offer.reuse_for(&picker.feature_request())`: `Some(vec)` feeds straight into
    /// [`pick`](Self::pick), `None` (or `!offer.satisfies(..)`) means run an own `zenanalyze`
    /// pass. The returned `Request` borrows `self`; drop it before the `&mut self`
    /// [`pick`](Self::pick).
    ///
    /// `None` for a model whose columns aren't all qualified (a pre-`name@hash` bake) — it
    /// can't reuse, so the caller runs its own pass. Requires the `api` feature.
    #[cfg(feature = "api")]
    #[must_use]
    pub fn feature_request(&self) -> Option<zenanalyze_api::Request<'_>> {
        self.wants
            .as_ref()
            .map(|w| zenanalyze_api::Request::new(zenanalyze_api::Select::Features(w)))
    }

    /// Borrow the underlying predictor — useful when the caller
    /// wants to read model metadata or run `predict` for diagnostics.
    pub fn predictor(&mut self) -> &mut Predictor<'b> {
        &mut self.predictor
    }

    /// Run argmin over the family dimension under the caller's
    /// allowed-family filter.
    ///
    /// `features` is the same feature vector the per-codec pickers
    /// consume — the bake declares which feature columns it uses
    /// (`feat_cols` in the manifest).
    ///
    /// Returns `Ok(None)` when every family is masked out (caller
    /// constraints unsatisfiable) and `Err` only on a runtime error
    /// (shape mismatch, NaN, …).
    ///
    /// **For per-family-score models only** (per-codec pickers, the gate, the lossless router).
    /// Refuses the **pairwise** lossy router ([`LOSSY_PAIRWISE_KEY`]) with
    /// [`PairwiseRouterNeedsRoute`](MetaPickerError::PairwiseRouterNeedsRoute): that model emits
    /// per-pair margins, so a raw argmin is meaningless — use [`route`](Self::route), which
    /// round-robins the margins into a family.
    pub fn pick(
        &mut self,
        features: &[f32],
        allowed: &AllowedFamilies,
    ) -> Result<Option<CodecFamily>, MetaPickerError> {
        // A pairwise lossy router emits per-pair margins, not per-family scores — a raw argmin
        // would silently mis-pick. Refuse it; the caller must use route() (which round-robins).
        if self
            .predictor
            .model()
            .metadata()
            .get_utf8(LOSSY_PAIRWISE_KEY)
            .is_ok()
        {
            return Err(MetaPickerError::PairwiseRouterNeedsRoute);
        }
        if !allowed.any() {
            return Ok(None);
        }
        let mask = AllowedMask::new(allowed.as_slice());
        let pick = self
            .predictor
            .argmin_masked(
                features,
                &mask,
                ScoreTransform::Identity,
                None::<&ArgminOffsets>,
            )
            .map_err(MetaPickerError::Predict)?;
        Ok(pick.map(|idx| CodecFamily::ALL[idx]))
    }

    /// The full quality-aware cross-codec route — the entry point.
    ///
    /// Composes the three router models (set via [`with_router`](Self::with_router)) with the
    /// content/latency masks into one decision:
    /// 1. narrow the family set: caller `allowed` ∩ [`content_capability`] (alpha/HDR rules
    ///    read from `offer`) ∩ [`viable`](AllowedFamilies::viable) (latency under `mode`);
    /// 2. **auto-gate** — lossy or lossless? An explicit [`QualityTarget::Lossless`] forces
    ///    lossless; otherwise the gate model decides from the features + target quality;
    /// 3. score the chosen branch's family router and [`resolve`](RouteDecision::resolve) a
    ///    masked argmin over the survivors (also intersected with the branch family set —
    ///    [`LOSSY`](AllowedFamilies::LOSSY) / [`LOSSLESS`](AllowedFamilies::LOSSLESS)).
    ///
    /// `offer` is a shared zenanalyze-api [`Offer`](zenanalyze_api::Offer); each model reuses
    /// its own feature columns from it, and the target quality is appended as the final input
    /// for the gate + lossy models (the lossless model takes no quality). `per_family_est_ms`
    /// is the codec's own per-family encode-time estimate (see
    /// [`viable`](AllowedFamilies::viable)).
    ///
    /// `Ok(None)` when no family survives the masks, or the offer can't satisfy a model's
    /// columns (the caller runs its own analysis pass); `Err` on a model/runtime error or a
    /// missing router model.
    #[cfg(feature = "api")]
    pub fn route(
        &mut self,
        offer: &zenanalyze_api::Offer<'_>,
        target: QualityTarget,
        allowed: AllowedFamilies,
        mode: zenpredict::EncodeMode,
        latency_ms: Option<u32>,
        per_family_est_ms: &[u32; CodecFamily::COUNT],
    ) -> Result<Option<RouteDecision>, MetaPickerError> {
        let allowed = allowed.intersect(content_capability(offer)).viable(
            mode,
            latency_ms,
            per_family_est_ms,
        );
        if !allowed.any() {
            return Ok(None);
        }
        // auto-gate: an explicit Lossless target bypasses the model
        let lossless = if target.is_lossless() {
            true
        } else {
            let g = score(
                self.gate
                    .as_mut()
                    .ok_or(MetaPickerError::RouterIncomplete)?,
                self.gate_wants.as_deref(),
                offer,
                Some(target.score_input()),
            )?;
            // `None` means the gate's columns aren't unqualified / the offer can't satisfy
            // them — per this fn's contract that's `Ok(None)` (the caller re-extracts), NOT a
            // silent default to the lossy branch. A wrong output count is a genuine
            // model/schema error, surfaced the same way the branch routers' shape check is.
            let Some(v) = g else {
                return Ok(None);
            };
            if v.len() < 2 {
                return Err(MetaPickerError::OutputShape {
                    expected: 2,
                    got: v.len(),
                });
            }
            v[1] < v[0] // [lossy, lossless], lower = better
        };
        let (branch, scored) = if lossless {
            (
                AllowedFamilies::LOSSLESS,
                score(
                    self.lossless
                        .as_mut()
                        .ok_or(MetaPickerError::RouterIncomplete)?,
                    self.lossless_wants.as_deref(),
                    offer,
                    None,
                )?,
            )
        } else {
            (
                AllowedFamilies::LOSSY,
                score(
                    &mut self.predictor,
                    self.wants.as_deref(),
                    offer,
                    Some(target.score_input()),
                )?,
            )
        };
        let Some(scored) = scored else {
            return Ok(None);
        };
        // Both family routers and the lossy pairwise router emit `CodecFamily::COUNT` outputs,
        // but they mean different things: the lossless router emits per-family scores
        // (lower = better) directly; the lossy router emits 6 pairwise margins (output `o` is
        // the oriented margin for `route::LOSSY_PAIRS[o]`). The lossy margins must be combined
        // by round-robin into per-family scores BEFORE the masked argmin. The gate + lossless
        // branches are unchanged.
        let n = if lossless {
            // The lossless (family) router's contract is one score per family.
            CodecFamily::COUNT
        } else {
            // The lossy router's contract is exactly the 6 pairs in LOSSY_PAIRS.
            route::LOSSY_PAIRS.len()
        };
        if scored.len() != n {
            return Err(MetaPickerError::OutputShape {
                expected: n,
                got: scored.len(),
            });
        }
        let scores = if lossless {
            let mut s = [f32::INFINITY; CodecFamily::COUNT];
            s.copy_from_slice(&scored);
            s
        } else {
            // 6 pairwise margins → per-family round-robin scores (lower = better; png/gif last).
            route::pairwise_round_robin(&scored)
        };
        Ok(RouteDecision::resolve(
            lossless,
            &scores,
            allowed.intersect(branch),
        ))
    }

    /// Read the [`FAMILY_ORDER_KEY`] (`zenpicker.family_order`)
    /// metadata key from the bake and confirm it matches
    /// [`ALL_LABELS_CSV`]. Returns `Ok(())` if the order matches,
    /// `Err` on mismatch (caller should refuse to use the picker —
    /// the bake was made against a different enum layout).
    ///
    /// Best practice: call once at startup, fail loudly on mismatch.
    pub fn validate_family_order(&mut self) -> Result<(), MetaPickerError> {
        let raw = self
            .predictor
            .model()
            .metadata()
            .get_utf8(FAMILY_ORDER_KEY)
            .map_err(|e| MetaPickerError::Metadata(format!("metadata: {:?}", e)))?;

        if raw == ALL_LABELS_CSV {
            Ok(())
        } else {
            Err(MetaPickerError::FamilyOrderMismatch {
                expected: ALL_LABELS_CSV.to_string(),
                actual: raw.to_string(),
            })
        }
    }
}

#[cfg(feature = "std")]
impl MetaPicker<'static> {
    /// The shipped default cross-codec router — three baked ZNPR models (lossy / lossless /
    /// auto-gate; wired 2026-06-30, retrained on zensim-A scoring 2026-07-01, 7f4d914) loaded
    /// from `include_bytes!` into process-static [`Model`]s, ready for [`route`](Self::route).
    /// Cheap to call repeatedly (the parsed models are cached in a `OnceLock`). Requires `std`
    /// for the static cache; `no_std` callers build their own via [`new`](Self::new) +
    /// [`with_router`](Self::with_router).
    ///
    /// All three are fit on the 101 qualified source-only zenanalyze features:
    /// - **lossy** — 6 pairwise linear discriminants over {jpeg, webp, jxl, avif} (each pair's
    ///   margin = a `LogisticRegression` projection of features + target_zq), combined
    ///   round-robin (`route::pairwise_round_robin`). f32 ZNPR (612 weights; exact margins).
    ///   Held-out RD overhead vs the perfect oracle **7.16% mean / 22.05% p90** (the prior
    ///   V0_2-scored bake measured 3.55% mean / 12.41% p90 — a real V0_2-vs-A metric shift, not
    ///   a picker regression; see 7f4d914). Fit: zenmetrics
    ///   `scripts/picker/pairwise_discriminants.py`.
    /// - **lossless / gate** — i8 family-score MLPs on the **support-aware (unbiased) oracle**
    ///   (only cells where all codecs have measured support; held-out 89.73% lossless / 98.18%
    ///   gate on zensim-A, matching/beating the prior V0_2 88.4% / 98.1%). Honest support thins
    ///   above ~zq88, so above there the gate defers to lossless.
    ///
    /// The offer passed to `route` must satisfy [`feature_request`](Self::feature_request)'s columns.
    pub fn default_routers() -> Self {
        use std::sync::OnceLock;
        static LOSSY: OnceLock<Model> = OnceLock::new();
        static LOSSLESS: OnceLock<Model> = OnceLock::new();
        static GATE: OnceLock<Model> = OnceLock::new();
        let lossy =
            LOSSY.get_or_init(|| Model::from_bytes(ROUTER_LOSSY).expect("baked lossy router"));
        let gate = GATE.get_or_init(|| Model::from_bytes(ROUTER_GATE).expect("baked gate router"));
        let lossless = LOSSLESS
            .get_or_init(|| Model::from_bytes(ROUTER_LOSSLESS).expect("baked lossless router"));
        MetaPicker::new(lossy).with_router(gate, lossless)
    }
}

/// The **blessed default** cross-codec route. Applies the shipped routers
/// ([`MetaPicker::default_routers`] — the f32 pairwise lossy router + the i8 gate/lossless MLPs) to
/// `offer` + `target`, **masked to the formats the caller can emit** (`available`). This is the
/// one call most consumers want:
///
/// ```rust,ignore
/// let decision = zenpicker::default_route(
///     &offer, QualityTarget::Zq(82.0),
///     &[CodecFamily::Jpeg, CodecFamily::Webp, CodecFamily::Avif], // what we support
///     EncodeMode::QueuedBalanced, None, &[0; CodecFamily::COUNT],  // no latency budget
/// )?;
/// ```
///
/// Equivalent to `MetaPicker::default_routers().route(offer, target,
/// AllowedFamilies::from_allowed(available), mode, latency_ms, per_family_est_ms)`. The format mask
/// composes with the content-capability mask (alpha/HDR) and the latency-`viable` mask inside
/// [`route`](MetaPicker::route).
///
/// For a hot loop, hold one [`MetaPicker::default_routers`] and call [`route`](MetaPicker::route)
/// repeatedly — this rebuilds the `Predictor` scratch each call (the parsed models are
/// process-static via `OnceLock`, so that part is free). `Ok(None)` when nothing `available` can
/// encode the image, or the offer lacks the routers' feature columns — fall back to [`family_rule`]
/// (the no-features, no-model prior).
#[cfg(all(feature = "std", feature = "api"))]
pub fn default_route(
    offer: &zenanalyze_api::Offer<'_>,
    target: QualityTarget,
    available: &[CodecFamily],
    mode: zenpredict::EncodeMode,
    latency_ms: Option<u32>,
    per_family_est_ms: &[u32; CodecFamily::COUNT],
) -> Result<Option<RouteDecision>, MetaPickerError> {
    MetaPicker::default_routers().route(
        offer,
        target,
        AllowedFamilies::from_allowed(available.iter().copied()),
        mode,
        latency_ms,
        per_family_est_ms,
    )
}

#[derive(Debug)]
#[non_exhaustive]
pub enum MetaPickerError {
    Predict(PredictError),
    Metadata(String),
    FamilyOrderMismatch {
        expected: String,
        actual: String,
    },
    /// [`route`](MetaPicker::route) was called without the gate/lossless models — call
    /// [`with_router`](MetaPicker::with_router) first.
    RouterIncomplete,
    /// A family router scored a wrong number of outputs (expected [`CodecFamily::COUNT`]).
    OutputShape {
        expected: usize,
        got: usize,
    },
    /// [`pick`](MetaPicker::pick) was called on a **pairwise** lossy router (carries
    /// [`LOSSY_PAIRWISE_KEY`]): its outputs are per-pair margins, not per-family scores, so a raw
    /// argmin is meaningless. Use [`route`](MetaPicker::route), which applies the round-robin.
    PairwiseRouterNeedsRoute,
}

#[cfg(feature = "std")]
impl core::fmt::Display for MetaPickerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Predict(e) => write!(f, "predict: {:?}", e),
            Self::Metadata(s) => write!(f, "metadata: {}", s),
            Self::FamilyOrderMismatch { expected, actual } => write!(
                f,
                "family order mismatch: bake declares {:?}, runtime expects {:?}",
                actual, expected
            ),
            Self::RouterIncomplete => {
                write!(
                    f,
                    "route() needs with_router (gate + lossless models) first"
                )
            }
            Self::OutputShape { expected, got } => write!(
                f,
                "router output shape: expected {} family scores, got {}",
                expected, got
            ),
            Self::PairwiseRouterNeedsRoute => write!(
                f,
                "pick() called on a pairwise lossy router (per-pair margins); use route()"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MetaPickerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn family_order_csv_matches_all() {
        let computed = CodecFamily::ALL
            .iter()
            .map(|f| f.label())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(computed, ALL_LABELS_CSV);
    }

    #[test]
    fn allowed_families_basic_ops() {
        let af = AllowedFamilies::none()
            .allow(CodecFamily::Jpeg)
            .allow(CodecFamily::Webp);
        assert!(af.is_allowed(CodecFamily::Jpeg));
        assert!(af.is_allowed(CodecFamily::Webp));
        assert!(!af.is_allowed(CodecFamily::Avif));
        assert!(af.any());

        let none = AllowedFamilies::none();
        assert!(!none.any());

        let all = AllowedFamilies::all();
        for fam in CodecFamily::ALL {
            assert!(all.is_allowed(fam));
        }
    }

    #[test]
    fn family_indexing_is_dense_and_zero_based() {
        for (i, fam) in CodecFamily::ALL.iter().enumerate() {
            assert_eq!(fam.index(), i);
        }
    }

    // per-family encode-cost (ms): jpeg/webp/png/gif fast, jxl/avif slow
    const EST_MS: [u32; CodecFamily::COUNT] = [10, 20, 200, 300, 5, 50];

    #[test]
    fn viable_realtime_masks_slow_families() {
        let v = AllowedFamilies::all().viable(
            zenpredict::EncodeMode::RealtimeFastest,
            Some(100),
            &EST_MS,
        );
        assert!(v.is_allowed(CodecFamily::Jpeg));
        assert!(v.is_allowed(CodecFamily::Webp));
        assert!(v.is_allowed(CodecFamily::Png));
        assert!(!v.is_allowed(CodecFamily::Jxl)); // 200ms > 100ms
        assert!(!v.is_allowed(CodecFamily::Avif)); // 300ms > 100ms
    }

    #[test]
    fn viable_queued_keeps_all_allowed() {
        let all = AllowedFamilies::all();
        assert_eq!(
            all.viable(zenpredict::EncodeMode::QueuedAggressive, Some(100), &EST_MS),
            all
        );
    }

    #[test]
    fn viable_no_ceiling_and_respects_prior_allowlist() {
        let all = AllowedFamilies::all();
        // no latency gate -> no masking
        assert_eq!(
            all.viable(zenpredict::EncodeMode::RealtimeFastest, None, &EST_MS),
            all
        );
        // prior allowlist {jpeg, jxl}, realtime 100ms: jpeg kept (fast), jxl dropped (slow),
        // webp never allowed
        let sub = AllowedFamilies::from_allowed([CodecFamily::Jpeg, CodecFamily::Jxl]);
        let v = sub.viable(zenpredict::EncodeMode::RealtimeFastest, Some(100), &EST_MS);
        assert!(v.is_allowed(CodecFamily::Jpeg));
        assert!(!v.is_allowed(CodecFamily::Jxl));
        assert!(!v.is_allowed(CodecFamily::Webp));
    }

    #[test]
    fn family_order_constants_are_consistent() {
        assert_eq!(CodecFamily::ALL.len(), CodecFamily::COUNT);
        assert_eq!(ALL_LABELS_CSV.split(',').count(), CodecFamily::COUNT);
    }

    #[test]
    fn family_order_key_is_zenpicker_namespaced() {
        // Reflects the rename: the meta-picker IS zenpicker now.
        // Keep this test so a future rename doesn't quietly drift.
        assert_eq!(FAMILY_ORDER_KEY, "zenpicker.family_order");
    }

    // The shipped default routers load from include_bytes! and route end-to-end. Builds an
    // offer satisfying the lossy model's 101 qualified columns and checks the gate behaviour.
    #[cfg(all(feature = "std", feature = "api"))]
    #[test]
    fn default_routers_load_and_route() {
        use alloc::string::{String, ToString};
        let mut r = MetaPicker::default_routers();
        let names: Vec<String> = r
            .predictor()
            .model()
            .feature_columns()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            names.len(),
            101,
            "lossy router expects 101 qualified feature columns"
        );
        let cells: Vec<zenanalyze_api::FeatureResult<'_>> = names
            .iter()
            .filter_map(|n| {
                zenanalyze_api::NamedFeature::parse(n)
                    .map(|nf| zenanalyze_api::FeatureResult::new(nf, 0.5f32))
            })
            .collect();
        assert_eq!(cells.len(), 101, "every column is a qualified name@hex8");
        let offer = zenanalyze_api::Offer::new(&cells, zenanalyze_api::Provenance::new("test"));
        let est = [0u32; CodecFamily::COUNT];
        let lossy = r
            .route(
                &offer,
                QualityTarget::Zq(85.0),
                AllowedFamilies::all(),
                zenpredict::EncodeMode::QueuedBalanced,
                None,
                &est,
            )
            .unwrap()
            .expect("satisfied offer routes");
        assert!(!lossy.lossless(), "zq85 -> lossy branch");
        let ll = r
            .route(
                &offer,
                QualityTarget::Lossless,
                AllowedFamilies::all(),
                zenpredict::EncodeMode::QueuedBalanced,
                None,
                &est,
            )
            .unwrap()
            .expect("lossless routes");
        assert!(ll.lossless(), "explicit Lossless -> lossless branch");
    }

    // ── Gate 1: round-trip the baked lossy router's MODEL MATH ─────────────────────────────
    //
    // Loads the shipped lossy router (`ROUTER_LOSSY` — the f32 6-pairwise-discriminant bake)
    // and runs `Predictor::predict` on a raw 102-input fixture vector, asserting the 6 raw
    // outputs equal the LogisticRegression margins the Python fit produced. This proves the
    // scaler (mean/scale), the standardized weights, and the row-major w[i*out+o] layout all
    // round-tripped through the bake correctly. (Note: predict() returns the RAW margins; the
    // sigmoid + round-robin live in `route::pairwise_round_robin`, exercised separately.)

    // Gate-1 round-trip fixture: a raw 102-input vector (101 features + target_zq last) and
    // the 6 expected pairwise margins = LogisticRegression.decision_function(scaler.transform(x)),
    // in LOSSY_PAIRS order. Generated by /tmp/fit_pairwise_router.py from the same fit that
    // produced the baked model. The baked model (identity layer over scaler-standardized
    // inputs) must reproduce these within 1e-3 (f32 weights → ~1e-6 in practice).
    const FIXTURE_INPUT: [f32; 102] = [
        1183.1138916015625f32,
        0.17489799857139587f32,
        0.17252899706363678f32,
        0.016346000134944916f32,
        0.007089999970048666f32,
        0.6805559992790222f32,
        0.3078700006008148f32,
        73.23847961425781f32,
        0.3379809856414795f32,
        1.2086269855499268f32,
        1077.0f32,
        0.0389540009200573f32,
        0.04207000136375427f32,
        0.02806999906897545f32,
        4.0f32,
        0.05073000118136406f32,
        0.032260000705718994f32,
        6.0f32,
        0.21217800676822662f32,
        2.953234910964966f32,
        19.560184478759766f32,
        12.18055534362793f32,
        0.004629999864846468f32,
        0.0f32,
        0.0f32,
        0.0f32,
        0.0f32,
        0.02510100044310093f32,
        2.4985430240631104f32,
        1.573091983795166f32,
        0.02196200005710125f32,
        0.012664999812841415f32,
        0.17592599987983704f32,
        0.05421699956059456f32,
        34.97286605834961f32,
        0.004629999864846468f32,
        0.07587400078773499f32,
        0.018518999218940735f32,
        0.0f32,
        27648.0f32,
        10.22730827331543f32,
        144.0f32,
        192.0f32,
        82944.0f32,
        0.75f32,
        0.2876819968223572f32,
        0.0f32,
        0.11111100018024445f32,
        3.0f32,
        1.6434530019760132f32,
        4.302288055419922f32,
        5.084190845489502f32,
        5.299294948577881f32,
        5.454172134399414f32,
        0.02708199992775917f32,
        0.049056001007556915f32,
        0.9158750176429749f32,
        1.0f32,
        0.021626999601721764f32,
        0.032552000135183334f32,
        0.22715100646018982f32,
        0.9389169812202454f32,
        1.0f32,
        9.0f32,
        66.0f32,
        191.0f32,
        255.0f32,
        0.0f32,
        0.0f32,
        0.0f32,
        0.15873000025749207f32,
        0.0f32,
        0.0f32,
        0.0f32,
        0.015873000025749207f32,
        10.22730827331543f32,
        10.22730827331543f32,
        10.332669258117676f32,
        0.0f32,
        0.0f32,
        0.0f32,
        0.7678509950637817f32,
        1.0314090251922607f32,
        1.1495100259780884f32,
        0.01309799961745739f32,
        0.018476000055670738f32,
        0.02196200005710125f32,
        0.0f32,
        0.0f32,
        10.304389953613281f32,
        0.7074009776115417f32,
        11.0f32,
        -0.03952400013804436f32,
        0.03018300049006939f32,
        1.6522489786148071f32,
        6.264340877532959f32,
        1.1378840208053589f32,
        1.0156769752502441f32,
        0.6501539945602417f32,
        1.335929036140442f32,
        1.0940029621124268f32,
        48.0f32,
    ];
    // Reference margins for the SHIPPED lossy pairwise router on FIXTURE_INPUT. Updated
    // 2026-07-01 when the router was retrained on the zensim-A (v47-strict-QAT) canonical
    // (the "ban V0_2 labels" re-bake). The values track whichever weights the .bin carries;
    // the test's purpose (bin margins round-trip through Predictor) is unchanged.
    const FIXTURE_MARGINS: [f32; 6] = [
        -3.30808f32,
        -3.4065585f32,
        -3.0315948f32,
        -0.71639067f32,
        -0.782362f32,
        -0.11527448f32,
    ];

    #[cfg(feature = "std")]
    #[test]
    fn lossy_router_roundtrip_matches_fixture_margins() {
        let model = Model::from_bytes(ROUTER_LOSSY).expect("baked lossy router parses");
        assert_eq!(model.n_inputs(), 102, "lossy router n_inputs");
        assert_eq!(model.n_outputs(), 6, "lossy router n_outputs");
        let mut pred = Predictor::new(&model);
        let out = pred.predict(&FIXTURE_INPUT).expect("predict");
        assert_eq!(out.len(), 6);
        for (i, (&got, &want)) in out.iter().zip(FIXTURE_MARGINS.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "pair {} margin: got {}, want {} (|diff|={})",
                i,
                got,
                want,
                (got - want).abs()
            );
        }
    }

    // pick() must REFUSE the pairwise lossy router: its 6 outputs are per-pair margins, not
    // per-family scores, so a raw argmin would silently mis-pick. The shipped lossy router carries
    // LOSSY_PAIRWISE_KEY; callers route() it instead (which round-robins the margins).
    #[test]
    fn pick_refuses_the_pairwise_lossy_router() {
        let model = Model::from_bytes(ROUTER_LOSSY).expect("baked lossy router parses");
        let mut mp = MetaPicker::new(&model);
        let err = mp
            .pick(&FIXTURE_INPUT, &AllowedFamilies::LOSSY)
            .expect_err("pick() on a pairwise router must error, not silently mis-pick");
        assert!(matches!(err, MetaPickerError::PairwiseRouterNeedsRoute));
    }
}
