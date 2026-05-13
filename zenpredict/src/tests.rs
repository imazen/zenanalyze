//! Crate-internal sanity checks — model parser, metadata parser,
//! argmin math. Round-trip tests that exercise bake → load → forward
//! live in `tests/roundtrip.rs`.

use crate::*;

#[test]
fn argmin_identity_no_offsets() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true, true, true, true, true];
    let m = AllowedMask::new(&mask);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, Some(1));
}

#[test]
fn argmin_respects_mask() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true, false, true, true, true];
    let m = AllowedMask::new(&mask);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, Some(3));
}

#[test]
fn argmin_empty_mask_returns_none() {
    let pred = [3.0f32, 1.0, 4.0];
    let mask = [false; 3];
    let m = AllowedMask::new(&mask);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, None);
}

#[test]
fn argmin_with_per_output_offsets_shifts_pick() {
    // Without offsets the lowest score is index 1 (=1.0).
    // With per_output adding +5 to index 1, the new lowest is 3 (=1.5).
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let offsets = ArgminOffsets {
        uniform: 0.0,
        per_output: Some(&[0.0, 5.0, 0.0, 0.0, 0.0]),
    };
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, Some(&offsets));
    assert_eq!(pick, Some(3));
}

#[test]
#[cfg(feature = "advanced")]
fn argmin_top_k_returns_sorted_indices() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let top = argmin::argmin_masked_top_k::<3>(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(top, [Some(1), Some(3), Some(0)]);
}

#[test]
#[cfg(feature = "advanced")]
fn pick_with_confidence_reports_gap() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let pick = argmin::pick_with_confidence(&pred, &m, ScoreTransform::Identity, None);
    let (idx, gap) = pick.unwrap();
    assert_eq!(idx, 1);
    assert!((gap - 0.5).abs() < 1e-6);
}

#[test]
#[cfg(feature = "advanced")]
fn pick_with_confidence_inf_when_only_one_allowed() {
    let pred = [3.0f32, 1.0, 4.0];
    let mask = [false, true, false];
    let m = AllowedMask::new(&mask);
    let (idx, gap) =
        argmin::pick_with_confidence(&pred, &m, ScoreTransform::Identity, None).unwrap();
    assert_eq!(idx, 1);
    assert!(gap.is_infinite());
}

#[test]
#[cfg(feature = "advanced")]
fn threshold_mask_finite_gate() {
    // INFINITY is non-finite → treated like NaN, gated out. The
    // mask gate is intentionally finite-only because `INFINITY` in
    // a reach-rate table conventionally means "missing data" not
    // "always reached."
    let rates = [0.99, 0.5, f32::NAN, 0.95, f32::INFINITY];
    let mut out = [false; 5];
    argmin::threshold_mask(&rates, 0.95, &mut out);
    assert_eq!(out, [true, false, false, true, false]);
}

#[test]
#[cfg(feature = "advanced")]
fn first_out_of_distribution_finds_first() {
    let bounds = [
        FeatureBound::new(0.0, 1.0),
        FeatureBound::new(-1.0, 1.0),
        FeatureBound::new(0.0, 100.0),
    ];
    assert_eq!(first_out_of_distribution(&[0.5, 0.0, 50.0], &bounds), None);
    assert_eq!(
        first_out_of_distribution(&[2.0, 0.0, 50.0], &bounds),
        Some(0)
    );
    assert_eq!(
        first_out_of_distribution(&[0.5, 0.0, f32::NAN], &bounds),
        Some(2)
    );
    assert_eq!(
        first_out_of_distribution(&[0.5, f32::INFINITY, 50.0], &bounds),
        Some(1)
    );
}

#[test]
fn metadata_empty_blob_yields_empty() {
    let m = Metadata::parse(&[]).unwrap();
    assert!(m.is_empty());
}

#[test]
fn argmin_nan_score_is_silently_skipped() {
    // NaN never compares less than a finite value, so a cell with
    // NaN score is never picked. Documented contract.
    let pred = [3.0_f32, f32::NAN, 1.0];
    let m = AllowedMask::new(&[true, true, true]);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, Some(2));
}

#[test]
fn argmin_all_nan_returns_none() {
    // If every allowed cell scores NaN, returns None — same as no
    // allowed cells. Callers needing the distinction must
    // pre-validate.
    let pred = [f32::NAN, f32::NAN, f32::NAN];
    let m = AllowedMask::new(&[true, true, true]);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, None);
}

#[test]
fn argmin_ties_prefer_lowest_index() {
    // Documented tie-break: lowest index wins (uses `<`, not `<=`).
    let pred = [5.0_f32, 5.0, 5.0];
    let m = AllowedMask::new(&[true, true, true]);
    assert_eq!(
        argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None),
        Some(0)
    );
}

#[test]
#[should_panic(expected = "mask.len()")]
fn argmin_short_mask_panics() {
    // mask.len() < predictions.len() panics in both debug and
    // release. Used to silently deny high-index cells, which masked
    // bugs.
    let pred = [3.0_f32, 1.0, 4.0];
    let m = AllowedMask::new(&[true, true]); // len 2 < 3
    let _ = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
}

#[test]
#[should_panic(expected = "mask.len()")]
#[cfg(feature = "advanced")]
fn argmin_with_scorer_short_mask_panics() {
    let m = AllowedMask::new(&[true, true]); // len 2 < n=3
    let _ = argmin::argmin_masked_with_scorer(3, &m, |i| i as f32);
}

#[test]
#[cfg(feature = "advanced")]
fn rescue_default_threshold_three_pp() {
    let policy = RescuePolicy::default();
    assert!((policy.rescue_threshold - 3.0).abs() < f32::EPSILON);
}


