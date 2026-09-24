//! CLI-only selection provenance. Existing library manifest structs stay intact.
use std::path::Path;

use serde::Serialize;
use zenpicker_train::{
    HeldoutManifest, PickerDataset, PickerEval, ScalarAxisSpec, TrainError,
    build_picker_dataset_with, build_picker_dataset_with_time_budget, file_sha256,
};

const OBJECTIVE: &str = "min_encoded_bytes_subject_to_quality_and_encode_time_budget";
const NOTE: &str = "Within each categorical cell, minimize encoded_bytes over rows with score_zensim >= target_zq and encode_time <= time_budget. Scalar targets use the same winning row. This constrains measured candidates; held-out actual encode timing is still required for model qualification.";

#[derive(Serialize)]
pub(super) struct TimeBudget {
    encode_time_column: String,
    time_budget_column: String,
}

impl TimeBudget {
    pub(super) fn from_columns(
        encode: Option<String>,
        budget: Option<String>,
    ) -> Result<Option<Self>, String> {
        match (encode, budget) {
            (None, None) => Ok(None),
            (Some(encode_time_column), Some(time_budget_column))
                if !encode_time_column.trim().is_empty() && !time_budget_column.trim().is_empty() =>
                Ok(Some(Self { encode_time_column, time_budget_column })),
            _ => Err("--encode-time-column and --time-budget-column require nonempty values and must be supplied together".into()),
        }
    }

    pub(super) fn annotate_training(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut value: toml::Table = toml::from_str(&std::fs::read_to_string(path)?)?;
        value.insert("formulation".into(), OBJECTIVE.into());
        value.insert("formulation_note".into(), NOTE.into());
        value.insert("dataset_selection".into(), toml::Value::try_from(self)?);
        std::fs::write(path, toml::to_string_pretty(&value)?)?;
        Ok(())
    }

    /// Evaluation and export have no existing training manifest to extend.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_record(
        &self,
        path: &Path,
        input: &Path,
        artifact: &Path,
        phase: &str,
        codec_filter: Option<&str>,
        val_frac: f64,
        ds: &PickerDataset,
        eval: Option<&PickerEval>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(Serialize)]
        struct Record<'a> {
            tool: &'static str,
            tool_version: &'static str,
            phase: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            codec_filter: Option<&'a str>,
            scalar_axes: &'a [String],
            formulation: &'static str,
            formulation_note: &'static str,
            input_parquet: String,
            input_sha256: String,
            artifact: String,
            artifact_sha256: String,
            val_frac: f64,
            picker_rows_total: usize,
            cell_labels: &'a [String],
            zq_targets: &'a [i64],
            dataset_selection: &'a TimeBudget,
            #[serde(skip_serializing_if = "Option::is_none")]
            heldout: Option<HeldoutManifest>,
        }
        let record = Record {
            tool: "zenpicker-train",
            tool_version: env!("CARGO_PKG_VERSION"),
            phase,
            codec_filter,
            scalar_axes: &ds.scalar_axes,
            formulation: OBJECTIVE,
            formulation_note: NOTE,
            input_parquet: input.display().to_string(),
            input_sha256: file_sha256(input)?,
            artifact: artifact.display().to_string(),
            artifact_sha256: file_sha256(artifact)?,
            val_frac,
            picker_rows_total: ds.n_rows(),
            cell_labels: &ds.cell_labels,
            zq_targets: &ds.zq_targets,
            dataset_selection: self,
            heldout: eval.map(heldout_manifest),
        };
        std::fs::write(path, toml::to_string_pretty(&record)?)?;
        eprintln!(
            "[zenpicker-train] wrote time-budget provenance: {}",
            path.display()
        );
        Ok(())
    }
}

pub(super) fn load_dataset(
    path: &Path,
    codec: Option<&str>,
    targets: &[i64],
    axes: &[ScalarAxisSpec],
    budget: Option<&TimeBudget>,
) -> Result<PickerDataset, TrainError> {
    match budget {
        Some(b) => build_picker_dataset_with_time_budget(
            path,
            codec,
            targets,
            axes,
            &b.encode_time_column,
            &b.time_budget_column,
        ),
        None => build_picker_dataset_with(path, codec, targets, axes),
    }
}

pub(super) fn heldout_manifest(eval: &PickerEval) -> HeldoutManifest {
    HeldoutManifest {
        bytes_srocc: eval.bytes_panel.srocc,
        bytes_plcc: eval.bytes_panel.plcc,
        bytes_krocc: eval.bytes_panel.krocc,
        bytes_pwrc: eval.bytes_panel.pwrc,
        bytes_z_rmse: eval.bytes_panel.z_rmse,
        bytes_or_ratio: eval.bytes_panel.or_ratio,
        argmin_acc: eval.argmin_acc,
        overhead_mean: eval.overhead_mean,
        overhead_p50: eval.overhead_p50,
        overhead_p90: eval.overhead_p90,
        n_rows: eval.n_rows,
        n_pairs: eval.n_pairs,
    }
}
