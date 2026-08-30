//! `evaluate_fixed_baselines` — the pre-declared "always-one-family"
//! baseline gate — on a hand-built dataset with known answers.
//!
//! Four rows over four cells in two families, chosen so every property the
//! gate rests on is pinned by an exact number: a family that reaches
//! everywhere, one that does not, a multi-cell family whose policy is its
//! own best reachable cell, a row where the fixed choice IS the oracle
//! (overhead exactly 0), and a row no cell reaches (never scored).

use zenpicker_train::{PickerDataset, evaluate_fixed_baselines};

const CELLS: [&str; 4] = ["a_lossy", "a_lossless", "b_lossy", "b_lossless"];

/// `bytes_log` is `ln(bytes)`; NaN where unreachable.
fn ds() -> PickerDataset {
    let ln = f64::ln;
    // row 0: all reach. oracle = a_lossy (100).
    // row 1: b_lossy unreachable. oracle = a_lossy (200).
    // row 2: family a unreachable entirely. oracle = b_lossy (400).
    // row 3: nothing reaches -> not scored at all.
    let rows: [[f64; 4]; 4] = [
        [100.0, 300.0, 150.0, 600.0],
        [200.0, 400.0, f64::NAN, 800.0],
        [f64::NAN, f64::NAN, 400.0, 500.0],
        [f64::NAN, f64::NAN, f64::NAN, f64::NAN],
    ];
    let mut bytes_log = Vec::new();
    let mut reach = Vec::new();
    for r in rows {
        for v in r {
            reach.push(v.is_finite());
            bytes_log.push(if v.is_finite() { ln(v) } else { f64::NAN });
        }
    }
    PickerDataset {
        features: vec![0.0; 4],
        n_in: 1,
        bytes_log,
        reach,
        n_cells: 4,
        cell_labels: CELLS.iter().map(|s| s.to_string()).collect(),
        image_ids: (0..4).map(|i| format!("img{i}")).collect(),
        target_zq: vec![80; 4],
        feature_names: vec!["feat_0".to_string()],
        zq_targets: vec![80],
        scalar_axes: Vec::new(),
        scalar_sentinels: Vec::new(),
        scalars: Vec::new(),
    }
}

fn get<'a>(
    v: &'a [zenpicker_train::FixedBaseline],
    label: &str,
) -> &'a zenpicker_train::FixedBaseline {
    v.iter()
        .find(|b| b.label == label)
        .unwrap_or_else(|| panic!("no baseline {label:?}"))
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn fixed_baselines_have_the_hand_computed_values() {
    let ds = ds();
    let out = evaluate_fixed_baselines(&ds, &[0, 1, 2, 3]);

    // Row 3 reaches nothing, so exactly 3 rows are scored — the same rule
    // `evaluate_picker_bake` uses.
    let scored = 3.0;

    // family:a = best reachable of {a_lossy, a_lossless}: 100, 200, none.
    // Overheads vs oracle (100, 200, 400): 0, 0. Covers 2 of 3 rows; it is
    // the oracle on both, so argmin_acc = 2/3 over ALL scored rows.
    let fa = get(&out, "family:a");
    assert_eq!(fa.n_rows, 2);
    assert!(close(fa.coverage, 2.0 / scored), "{}", fa.coverage);
    assert!(close(fa.overhead_mean, 0.0), "{}", fa.overhead_mean);
    assert!(close(fa.overhead_p90, 0.0));
    assert!(close(fa.argmin_acc, 2.0 / scored), "{}", fa.argmin_acc);

    // family:b = best reachable of {b_lossy, b_lossless}: 150, 800, 400.
    // Overheads: 150/100-1 = 0.5, 800/200-1 = 3.0, 400/400-1 = 0.
    // Full coverage; the oracle only on row 2.
    let fb = get(&out, "family:b");
    assert_eq!(fb.n_rows, 3);
    assert!(close(fb.coverage, 1.0));
    assert!(
        close(fb.overhead_mean, (0.5 + 3.0 + 0.0) / 3.0),
        "{}",
        fb.overhead_mean
    );
    assert!(close(fb.overhead_p50, 0.5), "{}", fb.overhead_p50);
    assert!(close(fb.argmin_acc, 1.0 / scored));

    // A per-cell policy is NOT the family policy when the family has more
    // than one cell: b_lossy reaches only rows 0 and 2.
    let cb = get(&out, "cell:b_lossy");
    assert_eq!(cb.n_rows, 2);
    assert!(close(cb.coverage, 2.0 / scored));
    assert!(
        close(cb.overhead_mean, (0.5 + 0.0) / 2.0),
        "{}",
        cb.overhead_mean
    );

    // Every cell and every family is reported, and nothing else.
    assert_eq!(out.len(), CELLS.len() + 2);
    for c in CELLS {
        let _ = get(&out, &format!("cell:{c}"));
    }
}

/// The family policy must be its family's OWN oracle (its best reachable
/// cell), which is the strongest fixed-family policy and therefore the most
/// conservative bar — never worse than any single cell of that family.
#[test]
fn family_policy_is_never_worse_than_its_own_cells() {
    let ds = ds();
    let out = evaluate_fixed_baselines(&ds, &[0, 1, 2, 3]);
    for fam in ["a", "b"] {
        let f = get(&out, &format!("family:{fam}"));
        for c in CELLS.iter().filter(|c| c.starts_with(fam)) {
            let cell = get(&out, &format!("cell:{c}"));
            assert!(
                f.coverage >= cell.coverage - 1e-12,
                "family:{fam} coverage {} < cell:{c} {}",
                f.coverage,
                cell.coverage
            );
            assert!(
                f.argmin_acc >= cell.argmin_acc - 1e-12,
                "family:{fam} argmin {} < cell:{c} {}",
                f.argmin_acc,
                cell.argmin_acc
            );
        }
    }
}

/// The oracle is the floor: no fixed policy can have a negative overhead.
#[test]
fn no_fixed_policy_beats_the_oracle() {
    let ds = ds();
    for b in evaluate_fixed_baselines(&ds, &[0, 1, 2, 3]) {
        if b.n_rows > 0 {
            assert!(
                b.overhead_mean >= -1e-12 && b.overhead_p50 >= -1e-12 && b.overhead_p90 >= -1e-12,
                "{} has a negative overhead: {:?}",
                b.label,
                (b.overhead_mean, b.overhead_p50, b.overhead_p90)
            );
        }
    }
}
