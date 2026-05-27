//! Emit a fitted picker model as a ZNPR v3 bake via the
//! `zenpredict-bake` JSON pipeline, plus a sibling TOML reproduce-this
//! manifest.
//!
//! Per zensim/CLAUDE.md "JSON pipeline mandate": we DO NOT hand-roll
//! the ZNPR wire format. We construct a `BakeRequestJson`-shaped JSON
//! document and run it through [`zenpredict_bake::bake_from_json_str`]
//! — the canonical Rust serializer. The output is ZNPR v3 (header
//! magic `ZNPR`, version byte `0x03`).
//!
//! Two model shapes are supported:
//!   - [`bake_ridge_to_znpr_v3`] — the single identity-layer ridge
//!     baseline (legacy / cheap reference).
//!   - [`bake_mlp_picker_to_znpr_v3`] — the real within-cell-optimal
//!     picker: an N-cell-output LeakyReLU MLP. `argmin(bytes_log,
//!     mask=reach)` over the outputs is the codec-config pick.

use std::fs;
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::TrainError;
use crate::mlp::{Mlp, MlpConfig};
use crate::model::RidgeModel;

/// Reproduce-this manifest for the ridge baseline bake.
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

/// Reproduce-this manifest for the within-cell-optimal MLP picker. The
/// fields capture EVERYTHING needed to regenerate the bake + judge it
/// honestly (formulation, features, model, search, held-out numbers,
/// data-coverage caveats).
#[derive(Debug, Serialize)]
pub struct MlpPickerManifest {
    pub tool: String,
    pub tool_version: String,
    /// `"within_cell_optimal_bytes_argmin"` — the zentrain formulation.
    pub formulation: String,
    pub formulation_note: String,
    pub codec_family: String,
    pub input_parquet: String,
    pub input_sha256: String,
    pub input_rows_total: usize,
    /// `(image, target_zq)` rows built (after the ceiling-aware skip).
    pub picker_rows_total: usize,
    pub train_rows: usize,
    pub val_rows: usize,
    /// Number of IMAGE feature columns (NOT counting the appended
    /// `zq_norm` input).
    pub n_image_features: usize,
    /// Total model inputs = image features + 1 (`zq_norm`).
    pub n_inputs: usize,
    /// Output dimension = number of categorical cells.
    pub n_cells: usize,
    pub cell_labels: Vec<String>,
    pub feature_names: Vec<String>,
    /// CONFIRMATION the codec's per-encode `q` is NOT an input.
    pub q_is_input: bool,
    pub inputs_note: String,
    /// `target_zq` grid (the REQUESTED-quality input axis).
    pub zq_targets: Vec<i64>,
    pub model: MlpModelManifest,
    pub search: SearchManifest,
    pub bake_sha256: String,
    pub bake_bytes: usize,
    pub znpr_version: u8,
    /// Held-out picker numbers (honest — no q-leakage).
    pub heldout: HeldoutManifest,
    /// Present only for distilled bakes (teacher → student). `None` for a
    /// direct hard-target fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distillation: Option<DistillManifest>,
    pub data_coverage_caveat: String,
    pub follow_ons: Vec<String>,
}

/// Provenance for teacher → student distillation (zentrain's recipe).
#[derive(Debug, Serialize, Clone)]
pub struct DistillManifest {
    /// `"histgb_per_cell_soft_target_mse"` — the zentrain recipe id.
    pub recipe: String,
    pub recipe_note: String,
    /// Teacher model family + hyperparameters.
    pub teacher_kind: String,
    pub teacher_max_iter: u32,
    pub teacher_max_depth: u32,
    pub teacher_learning_rate: f64,
    pub teacher_l2_regularization: f64,
    pub teacher_min_cell_rows: u32,
    pub teacher_random_state: u64,
    pub teacher_params_fingerprint: String,
    /// How many of the `n_cells` cells got a real per-cell teacher (vs the
    /// per-cell nanmean fallback for cells with < min_cell_rows reaching
    /// train rows).
    pub n_cells_with_teacher: usize,
    /// The student's distillation loss form.
    pub student_loss: String,
    /// Distillation blend weight on the soft target (1.0 = zentrain's
    /// pure soft-target MSE; < 1.0 mixed in hard targets where reachable).
    pub soft_weight: f64,
    /// Path + sha256 of the dataset export the teacher was fit on.
    pub teacher_dataset_export_path: String,
    pub teacher_dataset_export_sha256: String,
    /// Path + sha256 of the teacher's soft-target sidecar parquet.
    pub soft_targets_path: String,
    pub soft_targets_sha256: String,
    /// The Python teacher script that produced the soft targets.
    pub teacher_script: String,
    /// Teacher's OWN held-out argmin accuracy (the distillation ceiling),
    /// measured by the Python step on the val split.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub teacher_heldout_argmin_acc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub teacher_heldout_overhead_mean: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct MlpModelManifest {
    pub kind: String,
    pub hidden: Vec<usize>,
    pub activation: String,
    pub leaky_slope: f64,
    pub lr: f64,
    pub batch_size: usize,
    pub max_iter: usize,
    pub n_iter_no_change: usize,
    pub seed: u64,
    pub n_iter_ran: usize,
    pub best_internal_val_loss: f64,
    pub note: String,
}

#[derive(Debug, Serialize)]
pub struct SearchManifest {
    pub kind: String,
    /// Each candidate's `(hidden, lr, val_metric)` row, for provenance.
    pub candidates: Vec<SearchCandidate>,
    pub selected_index: usize,
    pub selection_metric: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct SearchCandidate {
    pub hidden: Vec<usize>,
    pub lr: f64,
    pub seed: u64,
    /// Held-out selection metric (bytes-log SROCC) used to rank.
    pub heldout_bytes_srocc: f64,
    pub argmin_acc: f64,
}

#[derive(Debug, Serialize)]
pub struct HeldoutManifest {
    /// SROCC of predicted-vs-actual `bytes_log` over reachable cells.
    pub bytes_srocc: f64,
    pub bytes_plcc: f64,
    pub bytes_krocc: f64,
    pub bytes_pwrc: f64,
    pub bytes_z_rmse: f64,
    pub bytes_or_ratio: f64,
    /// Fraction of held-out rows where the argmin pick == true best.
    pub argmin_acc: f64,
    pub overhead_mean: f64,
    pub overhead_p50: f64,
    pub overhead_p90: f64,
    pub n_rows: usize,
    pub n_pairs: usize,
}

/// Result of a successful bake: the bytes plus the paths written.
#[derive(Debug)]
pub struct BakeOutcome {
    pub bake_bytes: Vec<u8>,
    pub bake_path: String,
    pub manifest_path: String,
}

const SCHEMA_HASH: u64 = 0; // skeleton: no compiled-in schema gate yet.

/// Build the ZNPR v3 bake JSON for a ridge model. (Legacy baseline.)
pub fn bake_ridge_to_znpr_v3(
    model: &RidgeModel,
    feature_names: &[String],
    codec_family: &str,
    target_column: &str,
) -> Result<Vec<u8>, TrainError> {
    let p = model.n_features();
    let weights: Vec<f32> = model.weights.iter().map(|&w| w as f32).collect();
    let biases = vec![model.intercept as f32];
    let scaler_mean = model.standardizer.mean.clone();
    let scaler_scale = model.standardizer.scale.clone();

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
            { "key": "zenpicker.codec_family", "type": "utf8", "text": codec_family },
            { "key": "zenpicker.target_column", "type": "utf8", "text": target_column },
            { "key": "zenpicker_train.feature_names", "type": "utf8", "text": feature_names.join(",") },
            { "key": "zenpicker_train.model_kind", "type": "utf8", "text": "ridge_linear_baseline" },
        ],
    });

    let json =
        serde_json::to_string(&doc).map_err(|e| TrainError::Bake(format!("json encode: {e}")))?;
    let bytes =
        zenpredict_bake::bake_from_json_str(&json).map_err(|e| TrainError::Bake(e.to_string()))?;
    assert_eq!(&bytes[0..4], b"ZNPR", "bake must carry the ZNPR magic");
    assert_eq!(
        bytes[4], 0x03,
        "bake must be ZNPR v3 (header byte 4 = 0x03)"
    );
    Ok(bytes)
}

/// Build the ZNPR v3 bake JSON for the within-cell-optimal MLP picker.
///
/// The MLP maps standardized `[image_features, zq_norm]` →
/// `bytes_log[0..n_cells]`. The input standardizer (`scaler_mean` /
/// `scaler_scale`) is folded into the bake so the runtime applies it
/// for free. Layers: `n_in → h0 → … → n_cells`, LeakyReLU on hidden
/// layers, identity on the output layer. The codec runtime selects the
/// config via `argmin(output, mask=reach)`.
#[allow(clippy::too_many_arguments)]
pub fn bake_mlp_picker_to_znpr_v3(
    mlp: &Mlp,
    scaler_mean: &[f64],
    scaler_scale: &[f64],
    feature_names: &[String],
    cell_labels: &[String],
    codec_family: &str,
    zq_targets: &[i64],
    distilled: bool,
) -> Result<Vec<u8>, TrainError> {
    let baked = mlp.layers_for_bake();
    let n_layers = baked.len();
    let layers_json: Vec<serde_json::Value> = baked
        .iter()
        .enumerate()
        .map(|(i, (in_dim, out_dim, w, b))| {
            // LeakyReLU on hidden layers, identity on the final layer.
            let act = if i + 1 < n_layers {
                "leakyrelu"
            } else {
                "identity"
            };
            serde_json::json!({
                "in_dim": in_dim,
                "out_dim": out_dim,
                "activation": act,
                "dtype": "f32",
                "weights": w,
                "biases": b,
            })
        })
        .collect();

    let scaler_mean_f32: Vec<f32> = scaler_mean.iter().map(|&x| x as f32).collect();
    let scaler_scale_f32: Vec<f32> = scaler_scale.iter().map(|&x| x as f32).collect();
    let zq_csv = zq_targets
        .iter()
        .map(|z| z.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let doc = serde_json::json!({
        "schema_hash": SCHEMA_HASH,
        "flags": 0,
        "scaler_mean": scaler_mean_f32,
        "scaler_scale": scaler_scale_f32,
        "layers": layers_json,
        "metadata": [
            { "key": "zenpicker.codec_family", "type": "utf8", "text": codec_family },
            { "key": "zenpicker.formulation", "type": "utf8", "text": "within_cell_optimal_bytes_argmin" },
            { "key": "zenpicker_train.model_kind", "type": "utf8", "text": if distilled { "leakyrelu_mlp_picker_distilled" } else { "leakyrelu_mlp_picker" } },
            { "key": "zenpicker_train.training_target", "type": "utf8", "text": if distilled { "histgb_per_cell_soft_bytes_log" } else { "hard_within_cell_optimal_bytes_log" } },
            { "key": "zenpicker_train.image_feature_names", "type": "utf8", "text": feature_names.join(",") },
            // The full input order: image features then zq_norm.
            { "key": "zenpicker_train.input_order", "type": "utf8", "text": format!("{},zq_norm", feature_names.join(",")) },
            { "key": "zenpicker_train.cell_labels", "type": "utf8", "text": cell_labels.join("\n") },
            { "key": "zenpicker_train.zq_targets", "type": "utf8", "text": zq_csv },
        ],
    });

    let json =
        serde_json::to_string(&doc).map_err(|e| TrainError::Bake(format!("json encode: {e}")))?;
    let bytes =
        zenpredict_bake::bake_from_json_str(&json).map_err(|e| TrainError::Bake(e.to_string()))?;
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

/// Inputs the trainer threads into the ridge manifest at bake time.
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

/// Full ridge bake step (legacy baseline).
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

    let manifest_path = manifest_path_for(out_path);
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
            note: "legacy ridge baseline — single identity ZNPR layer; superseded by the \
                   within-cell-optimal MLP picker. See README."
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
            "Within-cell-optimal MLP picker (now the default path).".to_string(),
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

/// Everything the trainer threads into the MLP picker manifest.
pub struct MlpPickerManifestInputs<'a> {
    pub codec_family: &'a str,
    pub input_parquet: &'a str,
    pub input_sha256: &'a str,
    pub input_rows_total: usize,
    pub picker_rows_total: usize,
    pub train_rows: usize,
    pub val_rows: usize,
    pub n_image_features: usize,
    pub feature_names: &'a [String],
    pub cell_labels: &'a [String],
    pub zq_targets: &'a [i64],
    pub cfg: &'a MlpConfig,
    pub search: SearchManifest,
    pub heldout: HeldoutManifest,
    /// Distillation provenance, when this is a distilled bake.
    pub distillation: Option<DistillManifest>,
}

/// Full MLP picker bake step: serialize → write → sibling TOML manifest.
pub fn bake_mlp_picker(
    mlp: &Mlp,
    scaler_mean: &[f64],
    scaler_scale: &[f64],
    out_path: &Path,
    inputs: MlpPickerManifestInputs<'_>,
) -> Result<BakeOutcome, TrainError> {
    let bytes = bake_mlp_picker_to_znpr_v3(
        mlp,
        scaler_mean,
        scaler_scale,
        inputs.feature_names,
        inputs.cell_labels,
        inputs.codec_family,
        inputs.zq_targets,
        inputs.distillation.is_some(),
    )?;
    fs::write(out_path, &bytes)
        .map_err(|e| TrainError::Io(format!("write {}: {e}", out_path.display())))?;

    let manifest_path = manifest_path_for(out_path);
    let man = MlpPickerManifest {
        tool: "zenpicker-train".to_string(),
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        formulation: "within_cell_optimal_bytes_argmin".to_string(),
        formulation_note: "Ported from zentrain/tools/train_hybrid.py build_dataset: per \
            (image, target_zq), the target is bytes_log[cell] = ln(min encoded_bytes over \
            configs in that categorical cell whose score_zensim >= target_zq); reach[cell] \
            marks cells that hit the target. Pick = argmin(predicted bytes_log, mask=reach). \
            Categorical cells are zenjpeg's discrete knob combinations \
            (subsampling|progressive|sharp_yuv|effort) from knob_tuple_json."
            .to_string(),
        codec_family: inputs.codec_family.to_string(),
        input_parquet: inputs.input_parquet.to_string(),
        input_sha256: inputs.input_sha256.to_string(),
        input_rows_total: inputs.input_rows_total,
        picker_rows_total: inputs.picker_rows_total,
        train_rows: inputs.train_rows,
        val_rows: inputs.val_rows,
        n_image_features: inputs.n_image_features,
        n_inputs: mlp.n_in,
        n_cells: mlp.n_out,
        cell_labels: inputs.cell_labels.to_vec(),
        feature_names: inputs.feature_names.to_vec(),
        q_is_input: false,
        inputs_note: "Inputs are IMAGE features (feat_*) + zq_norm (the user's REQUESTED \
            target quality / 100). The codec's per-encode q is NOT an input — q is the \
            decision the picker makes. No q-leakage."
            .to_string(),
        zq_targets: inputs.zq_targets.to_vec(),
        model: MlpModelManifest {
            kind: if inputs.distillation.is_some() {
                "leakyrelu_mlp_picker_distilled".to_string()
            } else {
                "leakyrelu_mlp_picker".to_string()
            },
            hidden: inputs.cfg.hidden.clone(),
            activation: "leakyrelu".to_string(),
            leaky_slope: inputs.cfg.leaky_slope,
            lr: inputs.cfg.lr,
            batch_size: inputs.cfg.batch_size,
            max_iter: inputs.cfg.max_iter,
            n_iter_no_change: inputs.cfg.n_iter_no_change,
            seed: inputs.cfg.seed,
            n_iter_ran: mlp.n_iter,
            best_internal_val_loss: mlp.best_val_loss,
            note: "Matches zentrain's student topology (LeakyReLU(0.01), Adam, MSE, \
                   early-stopping) within Rust-port fidelity."
                .to_string(),
        },
        search: inputs.search,
        bake_sha256: sha256_hex(&bytes),
        bake_bytes: bytes.len(),
        znpr_version: 3,
        heldout: inputs.heldout,
        distillation: inputs.distillation.clone(),
        data_coverage_caveat: "The unified_v13_zenjpeg_cvvdp parquet sweeps only 5 q levels \
            {10,30,60,80,90} per image, so the 'reaches target_zq' ladder is COARSE. Per \
            zensim/CLAUDE.md 'Dense sampling for trained models', a production picker needs \
            ~30 q points + 16-20 log-spaced sizes. This bake validates the FORMULATION and \
            the Rust-vs-zentrain port; it is NOT a production picker. A dense size+quality \
            sweep is a separate data-gen task (documented follow-on)."
            .to_string(),
        follow_ons: vec![
            "Dense q (≈30 points) + log-spaced size sweep before any production bake.".to_string(),
            "Scalar prediction heads (chroma_scale/lambda) for sweeps that carry continuous \
             Pareto axes — this parquet's knobs are all categorical."
                .to_string(),
            "CubeCL GPU acceleration of the inner MLP training loop.".to_string(),
            "Cross-codec MetaPicker auto-regeneration when a per-codec bake updates.".to_string(),
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

fn manifest_path_for(out_path: &Path) -> std::path::PathBuf {
    let mut s = out_path.as_os_str().to_os_string();
    s.push(".toml");
    std::path::PathBuf::from(s)
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
