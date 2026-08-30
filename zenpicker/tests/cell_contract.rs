//! `CellContract` validation on synthetic bakes — the negative half.
//!
//! The real-bake tests (`metapicker_v1_contract.rs`) need a 104 KB artifact
//! from block storage and are skipped in CI. These bake tiny models in-process
//! with `zenpredict-bake`, so every refusal path the contract promises is
//! exercised on every CI run: a bad cell label, a cell count that disagrees
//! with `n_outputs`, a duplicate cell, an input order that is not a bijection
//! over the declared features plus exactly one `zq_norm`, and a declared width
//! that disagrees with the model.

use zenpicker::{AllowedFamilies, CellMode, CellPicker, CodecFamily, MetaPickerError};
use zenpredict_bake::bake_from_json_str;

/// One identity-ish layer, `n_in` → `n_out`, plus the three cell-contract
/// metadata keys verbatim (comma-separated lists, as `zenpicker-train` may
/// write them).
fn baked(n_in: usize, n_out: usize, cells: &str, feats: &str, order: &str) -> Vec<u8> {
    let weights: Vec<String> = (0..n_in * n_out)
        .map(|k| {
            // row-major [out][in]; a distinct small value per entry so no two
            // outputs tie by construction.
            let (o, i) = (k / n_in, k % n_in);
            format!("{:.3}", 1.0 + (o as f32) * 0.25 + (i as f32) * 0.0625)
        })
        .collect();
    let json = format!(
        r#"{{
        "schema_hash": 4242,
        "scaler_mean":  [{zeros}],
        "scaler_scale": [{ones}],
        "layers": [{{"in_dim": {n_in}, "out_dim": {n_out}, "activation": "identity",
                    "dtype": "f32", "weights": [{w}], "biases": [{obs}]}}],
        "metadata": [
            {{"key": "zenpicker_train.cell_labels",         "type": "utf8", "text": "{cells}"}},
            {{"key": "zenpicker_train.image_feature_names", "type": "utf8", "text": "{feats}"}},
            {{"key": "zenpicker_train.input_order",         "type": "utf8", "text": "{order}"}}
        ]
    }}"#,
        zeros = vec!["0.0"; n_in].join(","),
        ones = vec!["1.0"; n_in].join(","),
        w = weights.join(","),
        obs = vec!["0.0"; n_out].join(","),
    );
    bake_from_json_str(&json).unwrap()
}

/// A well-formed 2-feature (+`zq_norm`) / 2-cell contract.
fn ok_bake() -> Vec<u8> {
    baked(
        3,
        2,
        "zenjpeg_lossy,zenwebp_lossy",
        "a@11111111,b@22222222",
        "a@11111111,b@22222222,zq_norm",
    )
}

fn refusal(bytes: &[u8]) -> String {
    match CellPicker::from_znpr_bytes(bytes) {
        Ok(_) => panic!("this bake must be refused by the cell contract"),
        Err(MetaPickerError::CellContract(m)) => m,
        Err(other) => panic!("expected a CellContract error, got {other:?}"),
    }
}

#[test]
fn a_well_formed_cell_bake_loads_and_picks() {
    let bytes = ok_bake();
    let picker = CellPicker::from_znpr_bytes(&bytes).expect("well-formed cell bake loads");
    let c = picker.contract();

    assert_eq!(c.image_features(), ["a@11111111", "b@22222222"]);
    assert_eq!(c.input_order().len(), 3);
    assert_eq!(c.zq_index(), 2);
    assert_eq!(c.cells().len(), 2);
    assert_eq!(c.cells()[0].family(), CodecFamily::Jpeg);
    assert_eq!(c.cells()[0].mode(), CellMode::Lossy);
    assert_eq!(c.cells()[1].family(), CodecFamily::Webp);
    assert!(c.families().is_allowed(CodecFamily::Jpeg));
    assert!(c.families().is_allowed(CodecFamily::Webp));
    assert!(!c.families().is_allowed(CodecFamily::Avif));

    // the contract mapping reads each declared name once, places zq itself
    let mut asked = Vec::new();
    let x = c
        .build_input(0.9, |n| {
            asked.push(n.to_string());
            Some(match n {
                "a@11111111" => 1.0,
                "b@22222222" => 2.0,
                _ => panic!("out-of-contract read: {n}"),
            })
        })
        .expect("input");
    assert_eq!(asked, ["a@11111111", "b@22222222"]);
    assert_eq!(x, [1.0, 2.0, 0.9]);

    let pred = picker
        .predict_cells(&x, &c.families(), None)
        .expect("forward");
    assert_eq!(pred.scores().len(), 2);
    // weights are strictly increasing per output row, so cell 0 always scores
    // lower on a non-negative input ⇒ the argmin is cell 0.
    assert_eq!(pred.pick().map(|p| p.label()), Some("zenjpeg_lossy"));
    assert_eq!(pred.family(), Some(CodecFamily::Jpeg));

    // masks compose
    let webp_only = AllowedFamilies::none().allow(CodecFamily::Webp);
    assert_eq!(
        picker
            .predict_cells(&x, &webp_only, None)
            .expect("forward")
            .family(),
        Some(CodecFamily::Webp)
    );
    let reach = [false, true];
    assert_eq!(
        picker
            .predict_cells(&x, &c.families(), Some(&reach))
            .expect("forward")
            .pick()
            .map(|p| p.label()),
        Some("zenwebp_lossy")
    );
    assert!(
        picker
            .predict_cells(&x, &AllowedFamilies::none(), None)
            .expect("forward")
            .pick()
            .is_none()
    );
}

#[test]
fn missing_metadata_keys_are_refused() {
    // no zenpicker_train.* keys at all: a plain 2-in/2-out model.
    let json = r#"{
        "schema_hash": 1,
        "scaler_mean": [0.0, 0.0], "scaler_scale": [1.0, 1.0],
        "layers": [{"in_dim": 2, "out_dim": 2, "activation": "identity", "dtype": "f32",
                    "weights": [1.0,0.0, 0.0,1.0], "biases": [0.0, 0.0]}],
        "metadata": []
    }"#;
    let m = refusal(&bake_from_json_str(json).unwrap());
    assert!(m.contains("cell_labels"), "{m}");
}

#[test]
fn a_cell_label_that_is_not_family_mode_is_refused() {
    let m = refusal(&baked(
        3,
        2,
        "zenjpeg_lossy,zenheic_lossy", // heic is not a CodecFamily
        "a@11111111,b@22222222",
        "a@11111111,b@22222222,zq_norm",
    ));
    assert!(m.contains("zenheic_lossy"), "{m}");
}

#[test]
fn a_repeated_cell_is_refused() {
    let m = refusal(&baked(
        3,
        2,
        "zenjpeg_lossy,jpeg_lossy", // same (family, mode) written two ways
        "a@11111111,b@22222222",
        "a@11111111,b@22222222,zq_norm",
    ));
    assert!(m.contains("repeats"), "{m}");
}

#[test]
fn a_cell_count_that_disagrees_with_n_outputs_is_refused() {
    // 3 labels, 2 outputs — reading these as cells would index past the scores
    let m = refusal(&baked(
        3,
        2,
        "zenjpeg_lossy,zenwebp_lossy,zenavif_lossy",
        "a@11111111,b@22222222",
        "a@11111111,b@22222222,zq_norm",
    ));
    assert!(m.contains("cells but the model scores"), "{m}");
}

#[test]
fn an_input_order_that_disagrees_with_the_model_width_is_refused() {
    // model takes 3 inputs; the contract declares 2 names
    let m = refusal(&baked(
        3,
        2,
        "zenjpeg_lossy,zenwebp_lossy",
        "a@11111111",
        "a@11111111,zq_norm",
    ));
    assert!(m.contains("names but the model takes"), "{m}");
}

#[test]
fn an_input_order_without_zq_norm_is_refused() {
    let m = refusal(&baked(
        3,
        2,
        "zenjpeg_lossy,zenwebp_lossy",
        "a@11111111,b@22222222,c@33333333",
        "a@11111111,b@22222222,c@33333333",
    ));
    assert!(m.contains("zq_norm"), "{m}");
}

#[test]
fn an_input_order_that_repeats_a_feature_is_refused() {
    let m = refusal(&baked(
        3,
        2,
        "zenjpeg_lossy,zenwebp_lossy",
        "a@11111111,b@22222222",
        "a@11111111,a@11111111,zq_norm",
    ));
    // `b` is never placed and `a` is placed twice — either message is the bug
    assert!(m.contains("exactly once") || m.contains("appears"), "{m}");
}

#[test]
fn an_input_order_naming_an_undeclared_feature_is_refused() {
    let m = refusal(&baked(
        3,
        2,
        "zenjpeg_lossy,zenwebp_lossy",
        "a@11111111,b@22222222",
        "a@11111111,zzz@99999999,zq_norm",
    ));
    assert!(m.contains("zzz@99999999"), "{m}");
}

#[test]
fn zq_norm_declared_as_a_source_feature_is_refused() {
    let m = refusal(&baked(
        3,
        2,
        "zenjpeg_lossy,zenwebp_lossy",
        "a@11111111,zq_norm",
        "a@11111111,zq_norm,zq_norm",
    ));
    assert!(m.contains("quality input"), "{m}");
}

#[test]
fn a_repeated_source_feature_name_is_refused() {
    let m = refusal(&baked(
        3,
        2,
        "zenjpeg_lossy,zenwebp_lossy",
        "a@11111111,a@11111111",
        "a@11111111,a@11111111,zq_norm",
    ));
    assert!(m.contains("more than once"), "{m}");
}

#[test]
fn a_short_or_long_input_vector_is_refused_not_truncated() {
    let bytes = ok_bake();
    let picker = CellPicker::from_znpr_bytes(&bytes).unwrap();
    let fams = picker.contract().families();
    assert!(picker.predict_cells(&[1.0, 2.0], &fams, None).is_err());
    assert!(
        picker
            .predict_cells(&[1.0, 2.0, 0.9, 0.5], &fams, None)
            .is_err()
    );
    // a wrong-width reach mask too
    assert!(matches!(
        picker.predict_cells(&[1.0, 2.0, 0.9], &fams, Some(&[true])),
        Err(MetaPickerError::CellContract(_))
    ));
}

#[test]
fn the_schema_gate_runs_before_the_contract() {
    let bytes = ok_bake();
    assert!(CellPicker::from_znpr_bytes_with_schema(&bytes, 4242).is_ok());
    match CellPicker::from_znpr_bytes_with_schema(&bytes, 4243) {
        Err(MetaPickerError::Predict(_)) => {}
        Err(other) => panic!("a wrong schema hash must be a Predict error, got {other:?}"),
        Ok(_) => panic!("a wrong schema hash must be refused at load"),
    }
}
