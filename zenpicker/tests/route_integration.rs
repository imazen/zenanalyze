//! `MetaPicker::route` end-to-end: an auto-gate + lossy + lossless family router composed
//! over a shared zenanalyze-api `Offer`, with the content-capability + latency masks.
//! Three tiny baked models (linear, biases do the work) make the decisions predictable.
#![cfg(feature = "api")]

use zenanalyze_api::{FeatureResult, NamedFeature, Offer, Provenance};
use zenpicker::{AllowedFamilies, CodecFamily, MetaPicker, MetaPickerError, QualityTarget};
use zenpredict::{EncodeMode, Model};
use zenpredict_bake::bake_from_json_str;

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

// Two feature columns shared by all three models; the gate + lossy models take a 3rd input
// (the appended target quality).
const COLS: &str = "a@11111111\\nb@22222222";

fn bake(in_dim: usize, out_dim: usize, weights: &[f32], biases: &[f32]) -> Vec<u8> {
    let join = |v: &[f32]| {
        v.iter()
            .map(|x| format!("{x:?}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let mean = vec!["0.0"; in_dim].join(",");
    let scale = vec!["1.0"; in_dim].join(",");
    let json = format!(
        r#"{{
        "schema_hash": 1,
        "scaler_mean":  [{mean}],
        "scaler_scale": [{scale}],
        "layers": [{{"in_dim": {in_dim}, "out_dim": {out_dim}, "activation": "identity",
                    "dtype": "f32", "weights": [{w}], "biases": [{b}]}}],
        "metadata": [{{"key": "zentrain.feature_columns", "type": "utf8", "text": "{COLS}"}}]
    }}"#,
        w = join(weights),
        b = join(biases),
    );
    bake_from_json_str(&json).unwrap()
}

// gate: in = [a, b, target], out = [lossy_cost, lossless_cost]. out0 = target (lossy gets
// expensive with quality), out1 = 90 (flat). route picks lossless when out1 < out0, i.e.
// target > 90. Weight layout is [in_i -> out_o] row-major over inputs.
fn gate_bytes() -> Vec<u8> {
    let w = [
        0.0, 0.0, /*a*/ 0.0, 0.0, /*b*/ 1.0, 0.0, /*target -> out0*/
    ];
    bake(3, 2, &w, &[0.0, 90.0])
}
// lossy PAIRWISE router: in = [a,b,target], 6 outputs = the oriented margins for the 6 pairs
// route::LOSSY_PAIRS = [jpeg:webp, jpeg:jxl, jpeg:avif, webp:jxl, webp:avif, jxl:avif], in that
// order. Margin > 0 ⇒ the FIRST family of the pair wins (fewer bytes). `route()` runs these
// margins through round-robin (sum of per-pair win-probabilities) into per-family scores before
// the argmin. Weights are zero (margins = biases → feature-independent, predictable). These
// biases make jxl win all 3 of its pairs (→ score≈3), avif win 2 (→≈2), webp win 1 (→≈1),
// jpeg lose all (→≈0); round-robin order jxl > avif > webp > jpeg:
//   pair0 jpeg:webp = -8 → webp;  pair1 jpeg:jxl = -8 → jxl;  pair2 jpeg:avif = -8 → avif;
//   pair3 webp:jxl  = -8 → jxl;   pair4 webp:avif = -8 → avif; pair5 jxl:avif = +8 → jxl.
fn lossy_bytes() -> Vec<u8> {
    bake(3, 6, &[0.0; 18], &[-8.0, -8.0, -8.0, -8.0, -8.0, 8.0])
}
// lossless family router: in = [a,b], 6 outputs. among {webp,jxl,png}: jxl(1) < webp(4) < png(6).
fn lossless_bytes() -> Vec<u8> {
    bake(2, 6, &[0.0; 12], &[9.0, 4.0, 1.0, 9.0, 6.0, 9.0])
}
// Regression fixture for the output-shape bug: a lossless router baked with the WRONG number
// of outputs (3, not `CodecFamily::COUNT` == 6). `route()` must reject this with
// `MetaPickerError::OutputShape` rather than panicking in `copy_from_slice`.
fn lossless_bytes_wrong_count() -> Vec<u8> {
    bake(2, 3, &[0.0; 6], &[9.0, 4.0, 1.0])
}
// Regression fixture: a gate baked with the WRONG number of outputs (1, not the documented
// 2 = [lossy_cost, lossless_cost]). Prior bug: `score()` returning `Some(v)` with `v.len() < 2`
// fell into the same `_ => false` arm as "gate unavailable" and silently defaulted to lossy.
fn gate_bytes_wrong_count() -> Vec<u8> {
    bake(3, 1, &[0.0, 0.0, 0.0], &[0.0])
}

fn fr(q: &'static str, v: f32) -> FeatureResult<'static> {
    FeatureResult::new(NamedFeature::parse(q).unwrap(), v)
}

const NO_EST: [u32; CodecFamily::COUNT] = [0; CodecFamily::COUNT];

/// Run `f` with a fully-wired router (lossy primary + gate + lossless). Models live for the
/// closure; `MetaPicker` borrows them.
fn with_router<R>(f: impl FnOnce(&mut MetaPicker) -> R) -> R {
    let (g, ly, ll) = (
        Aligned(gate_bytes()),
        Aligned(lossy_bytes()),
        Aligned(lossless_bytes()),
    );
    let gm = Model::from_bytes(&g.0).unwrap();
    let lym = Model::from_bytes(&ly.0).unwrap();
    let llm = Model::from_bytes(&ll.0).unwrap();
    let mut r = MetaPicker::new(&lym).with_router(&gm, &llm);
    f(&mut r)
}

#[test]
fn lossy_target_routes_to_best_lossy_family() {
    with_router(|r| {
        // The pairwise margins (see lossy_bytes) make jxl win all 3 of its pairs → the
        // round-robin ranks jxl > avif > webp > jpeg.
        let feats = [fr("a@11111111", 1.0), fr("b@22222222", 2.0)];
        let offer = Offer::new(&feats, Provenance::new("t"));
        let d = r
            .route(
                &offer,
                QualityTarget::Zq(85.0),
                AllowedFamilies::all(),
                EncodeMode::QueuedBalanced,
                None,
                &NO_EST,
            )
            .unwrap()
            .unwrap();
        assert_eq!(d.family(), CodecFamily::Jxl);
        assert!(!d.lossless());
        assert_eq!(
            d.ranked(),
            &[
                CodecFamily::Jxl,
                CodecFamily::Avif,
                CodecFamily::Webp,
                CodecFamily::Jpeg
            ]
        );
    });
}

#[test]
fn near_lossless_target_gates_to_lossless() {
    with_router(|r| {
        let feats = [fr("a@11111111", 1.0), fr("b@22222222", 2.0)];
        let offer = Offer::new(&feats, Provenance::new("t"));
        // target 98 > 90 -> the gate flips to lossless; lossless router picks jxl.
        let d = r
            .route(
                &offer,
                QualityTarget::Zq(98.0),
                AllowedFamilies::all(),
                EncodeMode::QueuedBalanced,
                None,
                &NO_EST,
            )
            .unwrap()
            .unwrap();
        assert_eq!(d.family(), CodecFamily::Jxl);
        assert!(d.lossless());
        assert_eq!(
            d.ranked(),
            &[CodecFamily::Jxl, CodecFamily::Webp, CodecFamily::Png]
        );
    });
}

#[test]
fn explicit_lossless_bypasses_gate() {
    with_router(|r| {
        let feats = [fr("a@11111111", 1.0), fr("b@22222222", 2.0)];
        let offer = Offer::new(&feats, Provenance::new("t"));
        let d = r
            .route(
                &offer,
                QualityTarget::Lossless,
                AllowedFamilies::all(),
                EncodeMode::QueuedBalanced,
                None,
                &NO_EST,
            )
            .unwrap()
            .unwrap();
        assert!(d.lossless());
        assert_eq!(d.family(), CodecFamily::Jxl);
    });
}

#[test]
fn allowlist_narrows_the_lossy_pick() {
    with_router(|r| {
        let feats = [fr("a@11111111", 1.0), fr("b@22222222", 2.0)];
        let offer = Offer::new(&feats, Provenance::new("t"));
        // deny jxl + avif -> the round-robin's surviving families are webp (score≈1) and
        // jpeg (≈0), so webp wins.
        let allowed = AllowedFamilies::all()
            .deny(CodecFamily::Jxl)
            .deny(CodecFamily::Avif);
        let d = r
            .route(
                &offer,
                QualityTarget::Zq(85.0),
                allowed,
                EncodeMode::QueuedBalanced,
                None,
                &NO_EST,
            )
            .unwrap()
            .unwrap();
        assert_eq!(d.family(), CodecFamily::Webp);
    });
}

#[test]
fn alpha_capability_can_empty_the_set() {
    with_router(|r| {
        // alpha present denies jpeg; caller only allowed jpeg -> nothing survives -> None.
        let feats = [
            fr("a@11111111", 1.0),
            fr("b@22222222", 2.0),
            fr("alpha_present@00000000", 1.0),
        ];
        let offer = Offer::new(&feats, Provenance::new("t"));
        let allowed = AllowedFamilies::from_allowed([CodecFamily::Jpeg]);
        let d = r
            .route(
                &offer,
                QualityTarget::Zq(85.0),
                allowed,
                EncodeMode::QueuedBalanced,
                None,
                &NO_EST,
            )
            .unwrap();
        assert!(d.is_none());
    });
}

#[test]
fn wrong_lossless_output_count_returns_output_shape_error() {
    // Prior bug: the lossless-branch shape check compared `scored.len()` against itself
    // (`n = scored.len()` when `lossless`), which is tautological and can never fail, so a
    // lossless/gate model baked with `n_outputs != CodecFamily::COUNT` slipped past the check
    // and panicked in `copy_from_slice` inside `route()`. This must now return
    // `Err(MetaPickerError::OutputShape { expected: 6, got: 3 })` instead.
    let (g, ly, ll) = (
        Aligned(gate_bytes()),
        Aligned(lossy_bytes()),
        Aligned(lossless_bytes_wrong_count()),
    );
    let gm = Model::from_bytes(&g.0).unwrap();
    let lym = Model::from_bytes(&ly.0).unwrap();
    let llm = Model::from_bytes(&ll.0).unwrap();
    let mut r = MetaPicker::new(&lym).with_router(&gm, &llm);
    let feats = [fr("a@11111111", 1.0), fr("b@22222222", 2.0)];
    let offer = Offer::new(&feats, Provenance::new("t"));
    let err = r
        .route(
            &offer,
            QualityTarget::Lossless, // explicit lossless -> bypasses the gate model
            AllowedFamilies::all(),
            EncodeMode::QueuedBalanced,
            None,
            &NO_EST,
        )
        .unwrap_err();
    match err {
        MetaPickerError::OutputShape { expected, got } => {
            assert_eq!(expected, CodecFamily::COUNT);
            assert_eq!(got, 3);
        }
        other => panic!("expected MetaPickerError::OutputShape, got {other:?}"),
    }
}

#[test]
fn gate_wrong_output_count_returns_output_shape_error() {
    // Prior bug: a gate model scoring the wrong number of outputs (not 2) fell into the same
    // `_ => false` arm as "gate unavailable", silently defaulting to the lossy branch instead
    // of surfacing the schema error.
    let (g, ly, ll) = (
        Aligned(gate_bytes_wrong_count()),
        Aligned(lossy_bytes()),
        Aligned(lossless_bytes()),
    );
    let gm = Model::from_bytes(&g.0).unwrap();
    let lym = Model::from_bytes(&ly.0).unwrap();
    let llm = Model::from_bytes(&ll.0).unwrap();
    let mut r = MetaPicker::new(&lym).with_router(&gm, &llm);
    let feats = [fr("a@11111111", 1.0), fr("b@22222222", 2.0)];
    let offer = Offer::new(&feats, Provenance::new("t"));
    let err = r
        .route(
            &offer,
            QualityTarget::Zq(85.0), // non-lossless target -> exercises the gate model
            AllowedFamilies::all(),
            EncodeMode::QueuedBalanced,
            None,
            &NO_EST,
        )
        .unwrap_err();
    match err {
        MetaPickerError::OutputShape { expected, got } => {
            assert_eq!(expected, 2);
            assert_eq!(got, 1);
        }
        other => panic!("expected MetaPickerError::OutputShape, got {other:?}"),
    }
}

#[test]
fn gate_unsatisfiable_offer_returns_none_not_lossy_default() {
    // Prior bug: `score()` returning `None` (offer can't satisfy the gate's columns) was
    // treated the same as a real gate score of `[0.0, 0.0]` (default lossy) instead of
    // propagating the documented `Ok(None)` ("caller runs its own analysis pass") contract.
    let (g, ly, ll) = (
        Aligned(gate_bytes()),
        Aligned(lossy_bytes()),
        Aligned(lossless_bytes()),
    );
    let gm = Model::from_bytes(&g.0).unwrap();
    let lym = Model::from_bytes(&ly.0).unwrap();
    let llm = Model::from_bytes(&ll.0).unwrap();
    let mut r = MetaPicker::new(&lym).with_router(&gm, &llm);
    // Omit `b@22222222`, which the gate's own feature columns require -> `reuse_for` fails.
    let feats = [fr("a@11111111", 1.0)];
    let offer = Offer::new(&feats, Provenance::new("t"));
    let d = r
        .route(
            &offer,
            QualityTarget::Zq(85.0),
            AllowedFamilies::all(),
            EncodeMode::QueuedBalanced,
            None,
            &NO_EST,
        )
        .unwrap();
    assert!(
        d.is_none(),
        "unsatisfiable gate offer must be Ok(None), not a silent lossy default"
    );
}

#[test]
fn route_without_router_models_errors() {
    let ly = Aligned(lossy_bytes());
    let lym = Model::from_bytes(&ly.0).unwrap();
    let mut r = MetaPicker::new(&lym); // no with_router
    let feats = [fr("a@11111111", 1.0), fr("b@22222222", 2.0)];
    let offer = Offer::new(&feats, Provenance::new("t"));
    let err = r
        .route(
            &offer,
            QualityTarget::Zq(85.0),
            AllowedFamilies::all(),
            EncodeMode::QueuedBalanced,
            None,
            &NO_EST,
        )
        .unwrap_err();
    assert!(matches!(err, MetaPickerError::RouterIncomplete));
}
