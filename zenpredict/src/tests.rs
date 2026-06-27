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

// --- Opt-in `topk` surface: top-K query + compute-tier masking ---
//
// Gated behind `any(topk, advanced)` so they run under BOTH
// `--features topk` and the `advanced` superset, and stay off the
// default build. The whole point of the gate is that the default
// public API + monomorphization match a `predict` / `argmin_masked`-
// only consumer; these tests therefore only exist when the feature
// that adds the surface is enabled.
#[cfg(any(feature = "topk", feature = "advanced"))]
mod topk_surface {
    use super::*;

    #[test]
    fn stable_top_k_sorted_indices() {
        let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let mask = [true; 5];
        let m = AllowedMask::new(&mask);
        // Free function.
        let top = argmin::argmin_masked_top_k::<3>(&pred, &m, ScoreTransform::Identity, None);
        assert_eq!(top, [Some(1), Some(3), Some(0)]);
        // Re-exported at crate root under the same gate.
        let top2 = crate::argmin_masked_top_k::<2>(&pred, &m, ScoreTransform::Identity, None);
        assert_eq!(top2, [Some(1), Some(3)]);
    }

    #[test]
    fn stable_top_k_respects_mask_and_fills_none() {
        // Only 2 cells allowed but K=3 ⇒ third slot is None.
        let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let mask = [false, true, false, true, false];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k::<3>(&pred, &m, ScoreTransform::Identity, None);
        assert_eq!(top, [Some(1), Some(3), None]);
    }

    #[test]
    fn stable_top_k_in_range_indices_are_local() {
        // Sub-range [2..5] of the output; returned indices are within
        // the sub-range (0..3), not absolute.
        let pred = [9.0f32, 9.0, 4.0, 1.5, 3.0];
        let mask = [true; 3];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k_in_range::<2>(
            &pred,
            (2, 5),
            &m,
            ScoreTransform::Identity,
            None,
        );
        // sub-slice = [4.0, 1.5, 3.0] ⇒ argmin order 1 (1.5), 2 (3.0).
        assert_eq!(top, [Some(1), Some(2)]);
    }

    #[test]
    fn stable_top_k_in_range_out_of_bounds_all_none() {
        let pred = [1.0f32, 2.0, 3.0];
        let mask = [true; 2];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k_in_range::<2>(
            &pred,
            (2, 9), // end > len
            &m,
            ScoreTransform::Identity,
            None,
        );
        assert_eq!(top, [None, None]);
    }

    // --- tier_mask (compute-tier masking helper) ---

    #[test]
    fn tier_mask_admits_at_or_below_max() {
        // Boundary: tier == max_tier is ADMITTED.
        let tiers = [1u8, 3, 2, 3, 1];
        let mut out = [false; 5];
        crate::tier_mask(&tiers, 2, &mut out);
        assert_eq!(out, [true, false, true, false, true]);
    }

    #[test]
    fn tier_mask_zero_admits_only_cheapest() {
        let tiers = [0u8, 1, 0, 2];
        let mut out = [false; 4];
        crate::tier_mask(&tiers, 0, &mut out);
        assert_eq!(out, [true, false, true, false]);
    }

    #[test]
    fn tier_mask_high_max_admits_all() {
        let tiers = [0u8, 5, 9, 255];
        let mut out = [false; 4];
        crate::tier_mask(&tiers, 255, &mut out);
        assert_eq!(out, [true; 4]);
    }

    #[test]
    #[should_panic(expected = "tier_mask")]
    fn tier_mask_length_mismatch_panics() {
        let tiers = [1u8, 2, 3];
        let mut out = [false; 2]; // len 2 != 3
        crate::tier_mask(&tiers, 1, &mut out);
    }

    #[test]
    fn tier_mask_then_argmin_picks_cheapest_good() {
        // End-to-end shape: scores favor an expensive cell (idx 1,
        // score 1.0, tier 9) but the tier gate masks it out, so argmin
        // lands on the cheapest admissible cell (idx 3, score 1.5,
        // tier 1).
        let scores = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let tiers = [1u8, 9, 5, 1, 1];
        let mut gate = [false; 5];
        crate::tier_mask(&tiers, 2, &mut gate); // admit tiers <= 2
        let m = AllowedMask::new(&gate);
        let pick = argmin::argmin_masked(&scores, &m, ScoreTransform::Identity, None);
        assert_eq!(pick, Some(3));
    }

    // --- CELL_COMPUTE_TIER metadata key parses off the blob wire form ---
    //
    // `Model::cell_compute_tiers` is round-trip-tested through the bake
    // crate (`zenpredict-bake/tests/compute_tier.rs`); here we exercise
    // the underlying metadata-blob read it relies on, plus the
    // absent-key graceful-no-op contract.

    /// Encode one metadata entry in the v3 blob wire form:
    /// `key_len u8 | key | value_type u8 | value_len u32 LE | value`.
    fn push_md_entry(blob: &mut alloc::vec::Vec<u8>, key: &str, value_type: u8, value: &[u8]) {
        blob.push(key.len() as u8);
        blob.extend_from_slice(key.as_bytes());
        blob.push(value_type);
        blob.extend_from_slice(&(value.len() as u32).to_le_bytes());
        blob.extend_from_slice(value);
    }

    #[test]
    fn cell_compute_tier_key_round_trips_in_blob() {
        let tiers = [1u8, 3, 2, 9];
        let mut blob = alloc::vec::Vec::new();
        // value_type 0 = bytes (hot-path entries must be bytes, per the
        // metadata module contract).
        push_md_entry(&mut blob, keys::CELL_COMPUTE_TIER, 0, &tiers);
        let md = Metadata::parse(&blob).unwrap();
        let entry = md.get(keys::CELL_COMPUTE_TIER).expect("tier key present");
        assert_eq!(entry.value, &tiers[..]);

        // And it threads straight into the masking helper.
        let mut gate = [false; 4];
        crate::tier_mask(entry.value, 2, &mut gate);
        assert_eq!(gate, [true, false, true, false]);
    }

    #[test]
    fn cell_compute_tier_absent_is_graceful() {
        // A blob with a different key → looking up the tier key yields
        // None (which `Model::cell_compute_tiers` maps to an empty
        // slice).
        let mut blob = alloc::vec::Vec::new();
        push_md_entry(&mut blob, keys::BAKE_NAME, 1, b"some_picker_v1");
        let md = Metadata::parse(&blob).unwrap();
        assert!(md.get(keys::CELL_COMPUTE_TIER).is_none());
    }
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
