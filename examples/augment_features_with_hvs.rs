//! Augment an existing features TSV with the 5 HVS-derived features
//! (added 2026-05-17).
//!
//! Reads an existing features TSV, opens the source image referenced
//! by each row, resizes to the row's `(width, height)`, runs the
//! analyzer with the full `FeatureSet::SUPPORTED` (which under
//! `--features experimental` includes the HVS features), and emits a
//! new TSV with:
//!
//! - All original columns preserved verbatim
//! - Five new `feat_chroma_luma_covariance_cb`,
//!   `feat_chroma_luma_covariance_cr`, `feat_info_weight_mean`,
//!   `feat_info_weight_p90`, `feat_orientation_energy_ratio` columns
//!   appended at the end
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features experimental --example augment_features_with_hvs -- \
//!     --input  ~/work/zen/zenwebp/benchmarks/zenwebp_pareto_features_2026-05-01_combined.tsv \
//!     --output ~/work/zen/zenwebp/benchmarks/zenwebp_pareto_features_2026-05-17_hvs.tsv
//! ```
//!
//! Skip-existing: if the input already has any of the 5 HVS columns,
//! the row is passed through unmodified (idempotent re-runs are safe).

use image::ImageReader;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

const NEW_HVS_FEATURES: &[(&str, AnalysisFeature)] = &[
    ("feat_chroma_luma_covariance_cb", AnalysisFeature::ChromaLumaCovarianceCb),
    ("feat_chroma_luma_covariance_cr", AnalysisFeature::ChromaLumaCovarianceCr),
    ("feat_info_weight_mean", AnalysisFeature::InfoWeightMean),
    ("feat_info_weight_p90", AnalysisFeature::InfoWeightP90),
    ("feat_orientation_energy_ratio", AnalysisFeature::OrientationEnergyRatio),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                eprintln!("usage: --input <tsv> --output <tsv>");
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    let input = input.ok_or("missing --input")?;
    let output = output.ok_or("missing --output")?;

    let f = BufReader::new(File::open(&input)?);
    let mut lines = f.lines();
    let header_line = lines.next().ok_or("empty input")??;
    let header: Vec<&str> = header_line.split('\t').collect();
    let col_index: HashMap<&str, usize> = header
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    let image_path_idx = *col_index
        .get("image_path")
        .ok_or("input missing 'image_path' column")?;
    let width_idx = *col_index
        .get("width")
        .ok_or("input missing 'width' column")?;
    let height_idx = *col_index
        .get("height")
        .ok_or("input missing 'height' column")?;

    // Detect existing HVS columns to make this idempotent.
    let new_col_names: Vec<&str> = NEW_HVS_FEATURES.iter().map(|(n, _)| *n).collect();
    let existing_hvs: Vec<bool> =
        new_col_names.iter().map(|c| col_index.contains_key(c)).collect();
    if existing_hvs.iter().any(|&b| b) {
        eprintln!(
            "WARNING: input already has some HVS columns: {:?}",
            new_col_names
                .iter()
                .zip(&existing_hvs)
                .filter(|(_, b)| **b)
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
        );
    }

    let out = File::create(&output)?;
    let mut out = BufWriter::new(out);
    // Write header — preserve original, then append NEW HVS columns
    // (only the ones that aren't already present).
    let appended_cols: Vec<&str> = NEW_HVS_FEATURES
        .iter()
        .zip(&existing_hvs)
        .filter(|(_, exists)| !**exists)
        .map(|((name, _), _)| *name)
        .collect();
    let mut out_header = header.clone();
    out_header.extend_from_slice(&appended_cols);
    writeln!(out, "{}", out_header.join("\t"))?;

    let descriptor = PixelDescriptor::RGB8_SRGB;
    // Build the query — we only need the new HVS features, since all
    // other columns pass through from the input. Saves analyzer work.
    let mut feats = FeatureSet::default();
    for &(_, f) in NEW_HVS_FEATURES.iter() {
        feats = feats.with(f);
    }
    let query = AnalysisQuery::new(feats);

    let mut n_rows = 0usize;
    let mut n_errors = 0usize;
    let t_start = Instant::now();
    let mut last_log = Instant::now();
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != header.len() {
            eprintln!(
                "row {} has {} fields, header has {} — skipping",
                n_rows + 1,
                fields.len(),
                header.len()
            );
            n_errors += 1;
            continue;
        }
        let image_path = fields[image_path_idx];
        let width: u32 = fields[width_idx].parse().unwrap_or(0);
        let height: u32 = fields[height_idx].parse().unwrap_or(0);
        if width == 0 || height == 0 {
            n_errors += 1;
            continue;
        }

        let new_vals: Vec<f32> = match compute_hvs(image_path, width, height, &query, descriptor) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("row {}: failed on {} — {}", n_rows + 1, image_path, e);
                n_errors += 1;
                vec![f32::NAN; appended_cols.len()]
            }
        };

        // Write original row + appended values.
        let mut out_fields = fields.clone();
        let new_strs: Vec<String> =
            new_vals.iter().map(|v| format_f32(*v)).collect();
        for s in &new_strs {
            out_fields.push(s.as_str());
        }
        writeln!(out, "{}", out_fields.join("\t"))?;

        n_rows += 1;
        if last_log.elapsed().as_secs() >= 5 {
            let rate = n_rows as f64 / t_start.elapsed().as_secs_f64();
            eprintln!(
                "  ... {} rows ({:.1} rows/s, {} errors so far)",
                n_rows, rate, n_errors
            );
            last_log = Instant::now();
        }
    }
    let dt = t_start.elapsed();
    eprintln!(
        "wrote {} rows to {} in {:.1}s ({} errors)",
        n_rows,
        output.display(),
        dt.as_secs_f64(),
        n_errors
    );
    Ok(())
}

fn format_f32(v: f32) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else {
        format!("{:.6}", v)
    }
}

fn compute_hvs(
    image_path: &str,
    width: u32,
    height: u32,
    query: &AnalysisQuery,
    descriptor: PixelDescriptor,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let img = ImageReader::open(image_path)?.decode()?;
    let img = img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let bytes = rgb.into_raw();
    let stride = (w as usize) * 3;
    let slice = PixelSlice::new(&bytes, w, h, stride, descriptor)
        .map_err(|e| format!("PixelSlice::new: {e}"))?;
    let results = zenanalyze::analyze_features(slice, query)
        .map_err(|e| format!("analyze_features: {e}"))?;
    let mut out = Vec::with_capacity(NEW_HVS_FEATURES.len());
    for (_, feat) in NEW_HVS_FEATURES {
        let v = results.get_f32(*feat).unwrap_or(f32::NAN);
        out.push(v);
    }
    Ok(out)
}
