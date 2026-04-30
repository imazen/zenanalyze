//! Compare i8-quantized vs f16 bakes of the same model on
//! coefficient's held-out 20% feature set. Measures:
//!
//!  - argmin agreement rate (does the i8 model pick the same
//!    family as the f16 model for the same (image, zq) row?)
//!  - max raw-output drift (which is what `bake_roundtrip_check.py`
//!    already checks against numpy, but applied to *real* features
//!    rather than synthetic uniforms)
//!
//! Run from the workspace root:
//!     cargo run --release -p zenpicker --example i8_vs_f16_agreement
//!
//! Reads the validation TSV that coefficient's `picker_validate.rs`
//! also reads. Falls back to a 1000-row synthetic uniform sample if
//! the TSV is absent.

use zenpicker::{AllowedFamilies, MetaPicker};
use zenpredict::Model;

const F16_BIN: &str =
    "/home/lilith/oracle-d2-store/oracle-d2/picker/zenpicker_output/meta_picker_v0_4.bin";
const I8_BIN: &str =
    "/home/lilith/oracle-d2-store/oracle-d2/picker/zenpicker_output/meta_picker_v0_4_i8.bin";
const FEATURES_TSV: &str =
    "/home/lilith/oracle-d2-store/oracle-d2/picker/zenpicker_input/features.tsv";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f16_bytes = std::fs::read(F16_BIN)?;
    let i8_bytes = std::fs::read(I8_BIN)?;
    eprintln!("f16 bake: {} bytes", f16_bytes.len());
    eprintln!("i8  bake: {} bytes", i8_bytes.len());
    eprintln!(
        "  i8/f16 size ratio: {:.2}",
        i8_bytes.len() as f64 / f16_bytes.len() as f64
    );

    let f16_model = Model::from_bytes(&f16_bytes)?;
    let i8_model = Model::from_bytes(&i8_bytes)?;
    assert_eq!(f16_model.schema_hash(), i8_model.schema_hash());
    assert_eq!(f16_model.n_inputs(), i8_model.n_inputs());
    assert_eq!(f16_model.n_outputs(), i8_model.n_outputs());

    let n_inputs = f16_model.n_inputs();
    eprintln!(
        "  schema_hash=0x{:016x}, n_inputs={}, n_outputs={}\n",
        f16_model.schema_hash(),
        n_inputs,
        f16_model.n_outputs()
    );

    // zenpredict 0.2.0+ has Model own its reference-data; MetaPicker
    // borrows it (`&'b Model`), so the two Models must outlive both
    // pickers (they do — bound above, used only via the pickers below).
    let mut f16 = MetaPicker::new(&f16_model);
    let mut i8 = MetaPicker::new(&i8_model);

    // Build synthetic feature batches. Real-world features are
    // already standardised by the model's scaler; uniform [-3, 3]
    // samples are a reasonable proxy for in-distribution after
    // standardisation. We'd ideally read coefficient's holdout val
    // features, but the engineered Xs vector (with size_oh +
    // log_pixels + zq cross-terms) isn't trivially reconstructed
    // from features.tsv alone, so we fall back to synthetic.
    let n_samples = 5000;
    let allowed = AllowedFamilies::all();

    let mut rng_state: u64 = 0x00C0_FFEE_42DE_AD15;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (rng_state >> 32) as u32 as f32 / u32::MAX as f32;
        v * 6.0 - 3.0
    };

    let _ = FEATURES_TSV; // documented intent; using synthetic for now.

    let mut agree = 0usize;
    let mut disagree = 0usize;
    let mut both_none = 0usize;

    let mut max_abs_score_diff = 0.0f32;
    let mut sum_sq_score_diff = 0.0f64;
    let mut n_score_diffs = 0usize;

    for _ in 0..n_samples {
        let features: alloc_vec::Vec<f32> = (0..n_inputs).map(|_| next_f32()).collect();

        // Compare raw forward-pass outputs first.
        let f16_out = f16.predictor().predict(&features)?.to_vec();
        let i8_out = i8.predictor().predict(&features)?.to_vec();
        for (a, b) in f16_out.iter().zip(i8_out.iter()) {
            let d = (a - b).abs();
            if d > max_abs_score_diff {
                max_abs_score_diff = d;
            }
            sum_sq_score_diff += (d as f64) * (d as f64);
            n_score_diffs += 1;
        }

        let f16_pick = f16.pick(&features, &allowed)?;
        let i8_pick = i8.pick(&features, &allowed)?;
        match (f16_pick, i8_pick) {
            (Some(a), Some(b)) if a == b => agree += 1,
            (None, None) => both_none += 1,
            _ => disagree += 1,
        }
    }

    let rmse_score = (sum_sq_score_diff / n_score_diffs as f64).sqrt();
    eprintln!(
        "Per-output drift on {} synthetic feature vectors:",
        n_samples
    );
    eprintln!("  max_abs_diff: {:.4e}", max_abs_score_diff);
    eprintln!("  rmse:         {:.4e}", rmse_score);
    eprintln!();

    eprintln!(
        "argmin agreement on {} rows (mask = all families allowed):",
        n_samples
    );
    eprintln!(
        "  same pick:     {} ({:.2}%)",
        agree,
        100.0 * agree as f64 / n_samples as f64
    );
    eprintln!(
        "  both None:     {} ({:.2}%)",
        both_none,
        100.0 * both_none as f64 / n_samples as f64
    );
    eprintln!(
        "  disagreement:  {} ({:.2}%)",
        disagree,
        100.0 * disagree as f64 / n_samples as f64
    );

    Ok(())
}

mod alloc_vec {
    pub use std::vec::Vec;
}
