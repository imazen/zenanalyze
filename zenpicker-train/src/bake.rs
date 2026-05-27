//! Emit a fitted [`RidgeModel`] as a ZNPR v3 bake via the
//! `zenpredict-bake` JSON pipeline, plus a sibling TOML reproduce-this
//! manifest.
//!
//! Per zensim/CLAUDE.md "JSON pipeline mandate": we DO NOT hand-roll
//! the ZNPR wire format. We construct a [`BakeRequestJson`]-shaped
//! JSON document and run it through
//! [`zenpredict_bake::bake_from_json_str`] — the canonical Rust
//! serializer. The output is ZNPR v3 (header magic `ZNPR`, version
//! byte `0x03`).

use std::fs;
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::TrainError;
use crate::model::RidgeModel;

/// Reproduce-this manifest, mirrored on the §3 trainer's manifest
/// pattern: the full recipe needed to regenerate this exact bake.
#[derive(Debug, Serialize)]
pub struct PickerManifest {
    pub tool: String,
    pub tool_version: String,
    pub codec_family: String,
    pub target_column: String,
    pub input_parquet: String,
    pub input_sha256: String,
    pub input_rows_total: usize,
    pub train_rows: usize,
    pub val_rows: usize,
    pub n_features: usize,
    pub feature_names: Vec<String>,
    pub include_q_feature: bool,
    pub model: ModelManifest,
    pub bake_sha256: String,
    pub bake_bytes: usize,
    pub znpr_version: u8,
    /// Held-out panel numbers (recorded for provenance; the trainer
    /// also prints them).
    pub heldout_srocc: f64,
    pub heldout_plcc: f64,
    pub heldout_krocc: f64,
    pub heldout_n: usize,
    /// What this skeleton intentionally does NOT do yet.
    pub follow_ons: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelManifest {
    pub kind: String,
    pub lambda: f64,
    pub intercept: f64,
    pub note: String,
}

/// Result of a successful bake: the bytes plus the paths written.
#[derive(Debug)]
pub struct BakeOutcome {
    pub bake_bytes: Vec<u8>,
    pub bake_path: String,
    pub manifest_path: String,
}

const SCHEMA_HASH: u64 = 0; // skeleton: no compiled-in schema gate yet.

/// Build the ZNPR v3 bake JSON for a ridge model and serialize it via
/// the `zenpredict-bake` JSON pipeline. Returns the bake bytes.
///
/// Layout: one identity-activation F32 layer mapping `n_features → 1`.
/// The standardizer is folded into `scaler_mean` / `scaler_scale`;
/// the layer weights are the ridge weights (row-major `in_dim *
/// out_dim` = `n_features * 1`); the bias is the intercept.
pub fn bake_ridge_to_znpr_v3(
    model: &RidgeModel,
    feature_names: &[String],
    codec_family: &str,
    target_column: &str,
) -> Result<Vec<u8>, TrainError> {
    let p = model.n_features();

    // Weights: row-major in_dim × out_dim. out_dim = 1, so the column
    // vector of per-feature weights is the layer's flat weight slice.
    let weights: Vec<f32> = model.weights.iter().map(|&w| w as f32).collect();
    let biases = vec![model.intercept as f32];
    let scaler_mean = model.standardizer.mean.clone();
    let scaler_scale = model.standardizer.scale.clone();

    // Build the BakeRequestJson document. We emit JSON and run it
    // through bake_from_json_str — the JSON pipeline, not a hand-rolled
    // serializer.
    let doc = serde_json::json!({
        "schema_hash": SCHEMA_HASH,
        "flags": 0,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "layers": [
            {
                "in_dim": p,
                "out_dim": 1usize,
                "activation": "identity",
                "dtype": "f32",
                "weights": weights,
                "biases": biases,
            }
        ],
        "metadata": [
            {
                "key": "zenpicker.codec_family",
                "type": "utf8",
                "text": codec_family,
            },
            {
                "key": "zenpicker.target_column",
                "type": "utf8",
                "text": target_column,
            },
            {
                "key": "zenpicker_train.feature_names",
                "type": "utf8",
                "text": feature_names.join(","),
            },
            {
                "key": "zenpicker_train.model_kind",
                "type": "utf8",
                "text": "ridge_linear_baseline",
            },
        ],
    });

    let json =
        serde_json::to_string(&doc).map_err(|e| TrainError::Bake(format!("json encode: {e}")))?;
    let bytes =
        zenpredict_bake::bake_from_json_str(&json).map_err(|e| TrainError::Bake(e.to_string()))?;

    // Smoke-assert ZNPR v3 (per the v2-banned rule): magic + version.
    assert_eq!(&bytes[0..4], b"ZNPR", "bake must carry the ZNPR magic");
    assert_eq!(
        bytes[4], 0x03,
        "bake must be ZNPR v3 (header byte 4 = 0x03)"
    );

    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Full bake step: serialize the model to ZNPR v3, write it to
/// `out_path`, write a sibling `<out_path>.toml` manifest, and return
/// the outcome.
#[allow(clippy::too_many_arguments)]
pub fn bake_picker(
    model: &RidgeModel,
    feature_names: &[String],
    codec_family: &str,
    target_column: &str,
    out_path: &Path,
    manifest: PickerManifestInputs<'_>,
) -> Result<BakeOutcome, TrainError> {
    let bytes = bake_ridge_to_znpr_v3(model, feature_names, codec_family, target_column)?;

    fs::write(out_path, &bytes)
        .map_err(|e| TrainError::Io(format!("write {}: {e}", out_path.display())))?;

    let manifest_path = {
        let mut s = out_path.as_os_str().to_os_string();
        s.push(".toml");
        std::path::PathBuf::from(s)
    };

    let man = PickerManifest {
        tool: "zenpicker-train".to_string(),
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        codec_family: codec_family.to_string(),
        target_column: target_column.to_string(),
        input_parquet: manifest.input_parquet.to_string(),
        input_sha256: manifest.input_sha256.to_string(),
        input_rows_total: manifest.input_rows_total,
        train_rows: manifest.train_rows,
        val_rows: manifest.val_rows,
        n_features: model.n_features(),
        feature_names: feature_names.to_vec(),
        include_q_feature: feature_names.last().map(String::as_str) == Some("q"),
        model: ModelManifest {
            kind: "ridge_linear_baseline".to_string(),
            lambda: model.lambda,
            intercept: model.intercept,
            note: "skeleton baseline — single identity ZNPR layer; not SOTA. \
                   See README follow-ons for the mature non-linear search."
                .to_string(),
        },
        bake_sha256: sha256_hex(&bytes),
        bake_bytes: bytes.len(),
        znpr_version: 3,
        heldout_srocc: manifest.heldout_srocc,
        heldout_plcc: manifest.heldout_plcc,
        heldout_krocc: manifest.heldout_krocc,
        heldout_n: manifest.heldout_n,
        follow_ons: vec![
            "Full scikit-learn-parity hyperparameter search (cmaes/grid) — \
             this chunk ships a deterministic ridge baseline only."
                .to_string(),
            "CubeCL GPU acceleration of the inner training loop.".to_string(),
            "Cross-codec MetaPicker auto-regeneration when a per-codec bake updates.".to_string(),
            "Dense size/quality sampling per zensim/CLAUDE.md training-data discipline."
                .to_string(),
        ],
    };

    let toml =
        toml::to_string_pretty(&man).map_err(|e| TrainError::Bake(format!("toml encode: {e}")))?;
    fs::write(&manifest_path, toml)
        .map_err(|e| TrainError::Io(format!("write {}: {e}", manifest_path.display())))?;

    Ok(BakeOutcome {
        bake_bytes: bytes,
        bake_path: out_path.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
    })
}

/// The values the trainer threads into the manifest at bake time.
#[derive(Debug, Clone, Copy)]
pub struct PickerManifestInputs<'a> {
    pub input_parquet: &'a str,
    pub input_sha256: &'a str,
    pub input_rows_total: usize,
    pub train_rows: usize,
    pub val_rows: usize,
    pub heldout_srocc: f64,
    pub heldout_plcc: f64,
    pub heldout_krocc: f64,
    pub heldout_n: usize,
}

/// Compute the sha256 of a file's bytes (for the manifest's
/// `input_sha256`). Streams in 1 MiB chunks to bound memory.
pub fn file_sha256(path: &Path) -> Result<String, TrainError> {
    use std::io::Read;
    let mut f = fs::File::open(path)
        .map_err(|e| TrainError::Io(format!("open {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| TrainError::Io(format!("read {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    Ok(s)
}
