use zenpredict::*;
use zenpredict_bake::*;

#[cfg(test)]
mod safety_summary_tests {
    use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};
    use zenpredict::MetadataType;
    use zenpredict::{
        Activation, CellHint, FallbackEntry, Model, OutputBound, RescuePolicy, SafetyCompact,
        SafetyProfile, WeightDtype, keys, output_first_out_of_distribution,
    };
    use bytemuck::bytes_of;

    #[repr(C, align(16))]
    struct Aligned(Vec<u8>);

    fn synth_safety() -> SafetyCompact {
        let mut s = SafetyCompact {
            version: 1,
            flags: 0x0001,
            passed: 1,
            n_violations: 0,
            safety_profile: 0,
            threshold_set_version: 1,
            corpus_hash: 0xdeadbeef,
            mean_overhead_pct: 2.5,
            p99_shortfall_pp: 1.2,
            argmin_acc: 0.88,
            _reserved: [0; 8],
        };
        // Pack rescue_default + rescue_strict into _reserved[0..8].
        s._reserved[0..4].copy_from_slice(&3.0_f32.to_le_bytes());
        s._reserved[4..8].copy_from_slice(&1.0_f32.to_le_bytes());
        s
    }

    fn build_model_with_safety(
        safety: SafetyCompact,
        cell_hints: &[CellHint],
        fallback: &[FallbackEntry],
        output_bounds: &[OutputBound],
    ) -> Vec<u8> {
        let safety_bytes = bytes_of(&safety).to_vec();
        let cell_hint_bytes = bytemuck::cast_slice(cell_hints).to_vec();
        let fallback_bytes = bytemuck::cast_slice(fallback).to_vec();
        let output_bound_bytes = bytemuck::cast_slice(output_bounds).to_vec();

        let scaler_mean = [0.0f32];
        let scaler_scale = [1.0f32];
        let w = [1.0f32];
        let b = [0.0f32];

        let metadata = vec![
            BakeMetadataEntry {
                key: keys::SAFETY_COMPACT,
                kind: MetadataType::Numeric,
                value: &safety_bytes,
            },
            BakeMetadataEntry {
                key: keys::CELL_RESCUE_HINTS,
                kind: MetadataType::Bytes,
                value: &cell_hint_bytes,
            },
            BakeMetadataEntry {
                key: keys::ZQ_FALLBACK_TABLE,
                kind: MetadataType::Bytes,
                value: &fallback_bytes,
            },
            BakeMetadataEntry {
                key: keys::OUTPUT_BOUNDS,
                kind: MetadataType::Bytes,
                value: &output_bound_bytes,
            },
        ];

        let layers = [BakeLayer {
            in_dim: 1,
            out_dim: 1,
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
        })
        .unwrap()
    }

    #[test]
    fn safety_compact_round_trips_through_metadata() {
        let safety = synth_safety();
        let bytes = build_model_with_safety(safety, &[], &[], &[]);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let recovered = model.safety_compact().expect("safety_compact present");
        assert_eq!(recovered, safety);
        assert_eq!(recovered.profile(), Some(SafetyProfile::SizeOptimal));
        assert!(recovered.strict_certified());
    }

    #[test]
    fn cell_hints_round_trip() {
        let hints = [
            CellHint {
                suggested_strategy: 0,
                p99_shortfall_pp: 5,
                expected_rescue_rate: 100,
                flags: 0,
            },
            CellHint {
                suggested_strategy: 2,
                p99_shortfall_pp: 20,
                expected_rescue_rate: 200,
                flags: 0x04,
            },
        ];
        let bytes = build_model_with_safety(synth_safety(), &hints, &[], &[]);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let recovered = model.cell_rescue_hints();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0], hints[0]);
        assert_eq!(recovered[1], hints[1]);
        assert!(recovered[1].is_high_variance());
    }

    #[test]
    fn fallback_table_round_trips() {
        let table = [
            FallbackEntry {
                zq_target: 50,
                fallback_cell: 1,
                quality_bump_pp: 0,
                flags: 1,
            },
            FallbackEntry {
                zq_target: 85,
                fallback_cell: 3,
                quality_bump_pp: 5,
                flags: 1,
            },
        ];
        let bytes = build_model_with_safety(synth_safety(), &[], &table, &[]);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let recovered = model.zq_fallback_table();
        assert_eq!(recovered, &table[..]);
        assert_eq!(
            zenpredict::fallback_for(80.0, &recovered).map(|e| e.fallback_cell),
            Some(1) // 50 ≤ 80 < 85
        );
    }

    #[test]
    fn output_bounds_round_trip() {
        let bounds = [
            OutputBound::new(4.0, 12.0),
            OutputBound::new(0.0, 1.0),
            OutputBound::new(50.0, 95.0),
        ];
        let bytes = build_model_with_safety(synth_safety(), &[], &[], &bounds);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let recovered = model.output_bounds();
        assert_eq!(recovered, &bounds[..]);

        // OOD detection for sane inputs:
        assert_eq!(
            output_first_out_of_distribution(&[8.0, 0.5, 80.0], &recovered),
            None
        );
        // Picker hallucinated a zq=200 (impossible) — flagged.
        assert_eq!(
            output_first_out_of_distribution(&[8.0, 0.5, 200.0], &recovered),
            Some(2)
        );
        // NaN trips the gate.
        assert_eq!(
            output_first_out_of_distribution(&[f32::NAN, 0.5, 80.0], &recovered),
            Some(0)
        );
    }

    #[test]
    fn rescue_policy_from_bake_uses_safety_thresholds() {
        let safety = synth_safety(); // rescue_default = 3.0, rescue_strict = 1.0
        let bytes = build_model_with_safety(safety, &[], &[], &[]);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let pol_default = RescuePolicy::from_bake(&model, false);
        let pol_strict = RescuePolicy::from_bake(&model, true);
        assert!((pol_default.rescue_threshold - 3.0).abs() < 1e-6);
        assert!((pol_strict.rescue_threshold - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rescue_policy_falls_back_when_no_safety_compact() {
        // Build a model with no safety_compact metadata — RescuePolicy
        // should fall back to defaults rather than panic.
        let scaler_mean = [0.0f32];
        let scaler_scale = [1.0f32];
        let w = [1.0f32];
        let b = [0.0f32];
        let layers = [BakeLayer {
            in_dim: 1,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w,
            biases: &b,
        }];
        let bytes = bake(&BakeRequest {
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
        })
        .unwrap();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        assert!(model.safety_compact().is_none());
        let pol = RescuePolicy::from_bake(&model, false);
        // Should match Default (3.0).
        assert!((pol.rescue_threshold - 3.0).abs() < 1e-6);
    }

    #[test]
    fn empty_metadata_keys_return_empty_vecs() {
        // Model without any of the new keys — accessors return empty,
        // not error.
        let scaler_mean = [0.0f32];
        let scaler_scale = [1.0f32];
        let w = [1.0f32];
        let b = [0.0f32];
        let layers = [BakeLayer {
            in_dim: 1,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w,
            biases: &b,
        }];
        let bytes = bake(&BakeRequest {
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
        })
        .unwrap();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        assert!(model.cell_rescue_hints().is_empty());
        assert!(model.zq_fallback_table().is_empty());
        assert!(model.output_bounds().is_empty());
    }
}