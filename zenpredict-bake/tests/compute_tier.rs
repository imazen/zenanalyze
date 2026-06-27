//! Round-trip the optional `zentrain.cell_compute_tier` metadata key
//! through bake → load, and verify the generic `tier_mask` masking
//! helper threads off `Model::cell_compute_tiers`. The
//! absent-key-is-a-graceful-no-op contract (older bins still load) is
//! covered too.

#[cfg(test)]
mod compute_tier_tests {
    use zenpredict::{
        Activation, AllowedMask, Model, MetadataType, ScoreTransform, WeightDtype, argmin, keys,
        tier_mask,
    };
    use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

    #[repr(C, align(16))]
    struct Aligned(Vec<u8>);

    /// Bake a `n_out`-output identity-ish model, optionally carrying a
    /// `cell_compute_tier` table (empty `tiers` ⇒ key omitted).
    fn build_model(n_out: usize, tiers: &[u8]) -> Vec<u8> {
        // 1 input → n_out outputs, all weights 1.0 so output == input
        // broadcast; values don't matter for the metadata round-trip.
        let scaler_mean = [0.0f32];
        let scaler_scale = [1.0f32];
        let w = vec![1.0f32; n_out];
        let b = vec![0.0f32; n_out];

        let mut metadata: Vec<BakeMetadataEntry<'_>> = Vec::new();
        if !tiers.is_empty() {
            metadata.push(BakeMetadataEntry {
                key: keys::CELL_COMPUTE_TIER,
                // value_type 0 = bytes (hot-path entries must be bytes).
                kind: MetadataType::Bytes,
                value: tiers,
            });
        }

        let layers = [BakeLayer {
            in_dim: 1,
            out_dim: n_out,
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
            metadata: &metadata,
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

    #[test]
    fn cell_compute_tiers_round_trip() {
        let tiers = [1u8, 3, 2, 9, 1];
        let bytes = build_model(5, &tiers);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        assert_eq!(model.cell_compute_tiers(), &tiers[..]);
    }

    #[test]
    fn absent_tier_table_is_empty_no_op() {
        // Older-shaped bake with no tier key — accessor returns an
        // empty slice, not an error, and still loads.
        let bytes = build_model(3, &[]);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        assert!(model.cell_compute_tiers().is_empty());
    }

    #[test]
    fn tier_table_drives_a_fast_only_mask_then_argmin() {
        // End-to-end: the cheapest-scoring cell is expensive (tier 9);
        // a tier<=2 budget masks it out, so argmin lands on the
        // cheapest admissible cell.
        let tiers = [1u8, 9, 5, 1, 1];
        let bytes = build_model(5, &tiers);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();

        let model_tiers = model.cell_compute_tiers();
        assert_eq!(model_tiers, &tiers[..]);

        let scores = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let mut gate = vec![false; model_tiers.len()];
        tier_mask(model_tiers, 2, &mut gate);
        // tiers = [1, 9, 5, 1, 1], budget <=2 ⇒ idx 1 (9) and idx 2 (5)
        // masked out.
        assert_eq!(gate, vec![true, false, false, true, true]);

        let mask = AllowedMask::new(&gate);
        let pick = argmin::argmin_masked(&scores, &mask, ScoreTransform::Identity, None);
        // idx 1 (score 1.0) is tier 9 → masked; next-cheapest admissible
        // is idx 3 (score 1.5, tier 1).
        assert_eq!(pick, Some(3));

        // And the stable top-K query, also reachable here, returns the
        // ranked admissible candidates for an encode-verify pass.
        let top = model
            .metadata() // touch the model so the borrow is exercised
            .len();
        assert!(top >= 1); // tier key is present
        let topk = argmin::argmin_masked_top_k::<2>(&scores, &mask, ScoreTransform::Identity, None);
        assert_eq!(topk, [Some(3), Some(0)]); // 1.5 then 3.0
    }
}
