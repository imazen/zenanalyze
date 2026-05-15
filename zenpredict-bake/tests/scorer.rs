use zenpredict::*;
use zenpredict_bake::*;

#[cfg(test)]
mod scorer_tests {
    use super::*;

    #[test]
    fn standalone_scorer_finds_argmin_under_mask() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true, true, true, true, true];
        let mask = AllowedMask::new(&mask_data);
        let pick = argmin::argmin_masked_with_scorer(scores.len(), &mask, |i| scores[i]);
        assert_eq!(pick, Some(1));
    }

    #[test]
    fn standalone_scorer_respects_mask() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true, false, true, true, true];
        let mask = AllowedMask::new(&mask_data);
        let pick = argmin::argmin_masked_with_scorer(scores.len(), &mask, |i| scores[i]);
        assert_eq!(pick, Some(3));
    }

    #[test]
    fn standalone_scorer_empty_mask_returns_none() {
        let mask_data = [false; 5];
        let mask = AllowedMask::new(&mask_data);
        let pick = argmin::argmin_masked_with_scorer(5, &mask, |_| 0.0);
        assert_eq!(pick, None);
    }

    #[test]
    fn scorer_call_count_matches_allowed_indices() {
        // Closure mutates a side-effect counter; assert it fires
        // exactly once per allowed index.
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true, false, true, false, true];
        let mask = AllowedMask::new(&mask_data);
        use core::cell::Cell;
        let count = Cell::new(0);
        let _ = argmin::argmin_masked_with_scorer(scores.len(), &mask, |i| {
            count.set(count.get() + 1);
            scores[i]
        });
        assert_eq!(count.get(), 3, "scorer fired once per allowed index");
    }

    #[test]
    fn scorer_top_k_returns_sorted() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true; 5];
        let mask = AllowedMask::new(&mask_data);
        let top =
            argmin::argmin_masked_top_k_with_scorer::<3, _>(scores.len(), &mask, |i| scores[i]);
        assert_eq!(top, [Some(1), Some(3), Some(0)]);
    }

    #[test]
    fn scorer_top_k_with_fewer_allowed_pads_none() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [false, true, false, true, false];
        let mask = AllowedMask::new(&mask_data);
        let top =
            argmin::argmin_masked_top_k_with_scorer::<3, _>(scores.len(), &mask, |i| scores[i]);
        assert_eq!(top, [Some(1), Some(3), None]);
    }

    /// Identity-scorer should match `argmin_masked` with
    /// `ScoreTransform::Identity` and no offsets — sanity that the
    /// new path doesn't subtly diverge for the trivial case.
    #[test]
    fn scorer_identity_equivalent_to_argmin_masked_identity() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true, true, false, true, true];
        let mask = AllowedMask::new(&mask_data);
        let scorer_pick = argmin::argmin_masked_with_scorer(scores.len(), &mask, |i| scores[i]);
        let transform_pick = argmin::argmin_masked(&scores, &mask, ScoreTransform::Identity, None);
        assert_eq!(scorer_pick, transform_pick);
    }

    fn make_two_head_model() -> Vec<u8> {
        // Hybrid-heads layout: 3 cells, [bytes_log[0..3], time[3..6]].
        // Identity weights so the model passes its inputs straight
        // through to outputs (lets the test put exact known values
        // in the output without needing real training).
        use zenpredict_bake::{BakeLayer, BakeRequest, bake};
        let scaler_mean = [0.0_f32; 6];
        let scaler_scale = [1.0_f32; 6];
        let mut w = vec![0.0_f32; 6 * 6];
        for i in 0..6 {
            w[i * 6 + i] = 1.0;
        }
        let b = vec![0.0_f32; 6];
        let layers = [BakeLayer {
            in_dim: 6,
            out_dim: 6,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w,
            biases: &b,
        }];
        bake(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &[],
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            feature_order: None,
            output_order: None,
            compressed: false,
            hu_permutations: None,
        })
        .unwrap()
    }

    /// Motivating case: RD-vs-time argmin with hybrid-heads outputs.
    /// `score = bytes + μ·ms`; closure reads from both heads.

    #[test]
    fn predictor_scorer_rd_vs_time() {
        #[repr(C, align(16))]
        struct Aligned(Vec<u8>);
        let bytes = make_two_head_model();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(&model);

        // bytes_log = [10.0, 11.0, 12.0]    → bytes ≈ [22 026, 59 874, 162 754]
        // time      = [5.0, 8.0, 12.0]
        let features = [10.0_f32, 11.0, 12.0, 5.0, 8.0, 12.0];

        // μ = 0 → cell 0 (smallest bytes).
        let mask_data = [true; 3];
        let mask = AllowedMask::new(&mask_data);
        let pick = predictor
            .argmin_masked_with_scorer_in_range(&features, (0, 3), &mask, |out, i| {
                let bytes = out[i].exp();
                let ms = out[3 + i];
                bytes + 0.0 * ms
            })
            .unwrap();
        assert_eq!(pick, Some(0));

        // μ = 1e6 → time dominates; cell 0 still wins (5 ms vs 8/12).
        let pick = predictor
            .argmin_masked_with_scorer_in_range(&features, (0, 3), &mask, |out, i| {
                let bytes = out[i].exp();
                let ms = out[3 + i];
                bytes + 1.0e6 * ms
            })
            .unwrap();
        assert_eq!(pick, Some(0));

        // Skew the time axis: now cell 2 has the smallest time.
        let features2 = [10.0_f32, 11.0, 12.0, 100.0, 50.0, 1.0];
        // μ = 1e6 → time dominates → cell 2.
        let pick = predictor
            .argmin_masked_with_scorer_in_range(&features2, (0, 3), &mask, |out, i| {
                let bytes = out[i].exp();
                let ms = out[3 + i];
                bytes + 1.0e6 * ms
            })
            .unwrap();
        assert_eq!(pick, Some(2));
    }

    #[test]
    fn predictor_scorer_top_k_rd_vs_time() {
        #[repr(C, align(16))]
        struct Aligned(Vec<u8>);
        let bytes = make_two_head_model();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(&model);

        // bytes_log[0..3] = [10.0, 11.0, 12.0]
        // time[3..6]       = [100.0, 50.0, 1.0]
        let features = [10.0_f32, 11.0, 12.0, 100.0, 50.0, 1.0];

        let mask_data = [true; 3];
        let mask = AllowedMask::new(&mask_data);

        // μ = 1e6 → time dominates: cell 2 (time=1) < cell 1 (50) < cell 0 (100).
        let top = predictor
            .argmin_masked_top_k_with_scorer_in_range::<3, _>(&features, (0, 3), &mask, |out, i| {
                let bytes = out[i].exp();
                let ms = out[3 + i];
                bytes + 1.0e6 * ms
            })
            .unwrap();
        // First slot is cell 2; second is cell 1.
        assert_eq!(top[0], Some(2));
        assert_eq!(top[1], Some(1));
    }
}

// =====================================================================
// Issue: safety + rescue summary + output bounds (this PR).
// Verifies the new structs round-trip through the bake / load cycle
// via metadata, and that Model / RescuePolicy / Predictor expose
// them correctly.
// =====================================================================
