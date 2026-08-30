//! **Does the DECISION change?** — the question that actually decides whether
//! analysis thoroughness should scale with image size.
//!
//! `examples/budget_drift_grid` measures how far feature *values* move between
//! the crate-invariant sampling budgets and full sampling. Drift only matters if
//! it changes an output, so this example runs the real consumers over both
//! regimes and counts how often the pick differs.
//!
//! ## What is actually exercised
//!
//! | consumer | how it is driven here | fidelity |
//! |---|---|---|
//! | cross-codec meta-picker | `zenpicker::default_route` — the **real shipped entry point**, with the **real baked routers**, both in this workspace | exact |
//! | zenjpeg `encode/picker.rs` | its bake + `feature_order.txt` replayed through `zenpredict`, mirroring `run_model` | mirror |
//! | zenwebp classifier | its live threshold rules replayed on the 4 features that still reach a decision | mirror |
//!
//! The meta-picker path calls production code, so it cannot drift from what
//! ships. The other two are **mirrors**: this crate cannot depend on zenjpeg or
//! zenwebp (they depend on it), so the decision rule is re-stated here against a
//! cited source line. Each mirror asserts what it can about its source artifact
//! (feature count, cell count) and names the file it mirrors, so a change over
//! there shows up as a failed assert rather than a silently stale number.
//!
//! zenavif's `auto_tune` is **not** covered: its bake wants 43 columns of which
//! several (`text_likelihood`, `screen_content_likelihood`, `natural_likelihood`,
//! `line_art_score`, `log_min_dim`) no longer exist in this build and are
//! zero-filled by the real code, and it expands to a 96-dim engineered vector
//! plus two JSON LUTs. Mirroring that faithfully is more likely to measure my
//! transcription than its behaviour, so it is left out rather than guessed at.
//!
//! ## The A/B
//!
//! For each image cell, features are extracted twice through the same
//! `__analyze_internal` backdoor the drift grid uses — once at the shipping
//! budgets, once fully sampled — and every consumer is asked for a decision on
//! each. A row is emitted per (consumer, cell, target), with both decisions and
//! whether they differ.
//!
//! Targets sweep the quality dial at equal density across the whole range —
//! `zq` 5..97 step 4 — because the low-q regime is where picks are most
//! contested and a high-q-dense grid would understate the change rate.
//!
//! ```text
//! nice -n 19 cargo run --release --features hdr,api --example budget_decision_ab
//! ```
//!
//! Env: `ZENANALYZE_CORPUS_DIR`, `AB_SIDES`, `AB_CLASSES`, `AB_CROPS` (default 8),
//! `AB_OUT`, `ZENJPEG_PICKER_DIR` (default `../zenjpeg/zenjpeg/src/encode/picker_data`).
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use zenanalyze::feature::{
    AnalysisFeature, AnalysisQuery, AnalysisResults, FeatureSet, FeatureValue,
};
use zenanalyze::versioning::feature_version_hash;
use zenanalyze_api::{NamedFeature, Offer, OwnedFeatureResult, Provenance, Value};
use zenpixels::{PixelDescriptor, PixelSlice};
use zenpredict::{
    AllowedMask, EncodeMode, Model, Predictor, ScoreTransform, argmin_masked_in_range,
};

#[path = "common/mod.rs"]
mod common;
use common::{ALL_SIDES, Class, build, classes, extra_classes};

const HF_UNCAPPED: usize = 262_144;
const DEFAULT_PIXEL_BUDGET: usize = 500_000;
const DEFAULT_HF_MAX_BLOCKS: usize = 1_024;

/// Quality dial sweep. Equal density across the range on purpose — a grid denser
/// at high q would understate the change rate exactly where picks are contested.
fn targets() -> Vec<f32> {
    (0..24).map(|i| 5.0 + 4.0 * i as f32).collect()
}

fn analyze(buf: &[u8], side: u32, pb: usize, hf: usize) -> AnalysisResults {
    let iq = AnalysisQuery::__internal_with_overrides(FeatureSet::SUPPORTED, pb, hf);
    let slice = PixelSlice::new(
        buf,
        side,
        side,
        side as usize * 3,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap();
    zenanalyze::__analyze_internal(slice, &iq).expect("analysis")
}

/// Build the qualified-name cells an `Offer` needs from an already-computed
/// result. Mirrors `src/offer.rs`'s `extract_offer` body (lines 61-75); it
/// cannot be reused directly because that function runs its own analysis pass at
/// the default budgets, which is the very thing being varied.
///
/// # The version bridge, and why it is here
///
/// All three shipped routers were baked against `chroma_subsample_dct_loss@48f0f976`;
/// this build produces `@fabc9776`. That single drifted column makes every
/// `Select::Features` want miss, so `default_route` returns `Ok(None)` for
/// **every** offer from this build — no route at all, at any target, on any
/// image. (That is a real finding about the shipped routers, reported separately;
/// it is not caused by sampling budgets.)
///
/// To measure anything about routing, this harness re-qualifies that one column
/// to the hash the routers expect. **That is precisely the silent substitution
/// `Select::Features` exists to prevent, done deliberately, in a measurement
/// harness, and it must never be copied into production code.** It is sound
/// *here* only because it is applied identically to both arms, so it cannot
/// manufacture or mask a difference between them — it only makes the router
/// answer at all. Set `AB_NO_BRIDGE=1` to disable it and observe the universal
/// `none`.
fn offer_cells(res: &AnalysisResults, bridge: bool) -> Vec<OwnedFeatureResult> {
    let mut cells = Vec::new();
    for f in FeatureSet::SUPPORTED.iter() {
        let (Some(version), Some(value)) = (feature_version_hash(f), res.get(f)) else {
            continue;
        };
        let mut qualified = NamedFeature::qualified_for(f.name(), NamedFeature::fold_hash(version));
        if bridge && f.name() == "chroma_subsample_dct_loss" {
            qualified = "chroma_subsample_dct_loss@48f0f976".to_string();
        }
        let v = match value {
            FeatureValue::F32(x) => Value::F32(x),
            FeatureValue::U32(x) => Value::U32(x),
            FeatureValue::U64(x) => Value::U64(x),
            FeatureValue::Bool(b) => Value::Bool(b),
            // `FeatureValue` is `#[non_exhaustive]`. A variant added later would
            // be silently dropped from the offer, which is the safe direction:
            // the consumer misses that column and re-extracts rather than
            // getting a guessed conversion.
            _ => continue,
        };
        cells.push(OwnedFeatureResult::new(&qualified, v));
    }
    cells
}

// ---------------------------------------------------------------- zenjpeg
//
// MIRROR of `zenjpeg/zenjpeg/src/encode/picker.rs` (`run_model`, `cell_to_config`,
// `build_inputs_from_results`, `resolve_features`) as of 2026-08-30. Constants
// asserted below so a change to the bake trips an assert instead of quietly
// changing the meaning of the measurement.

const ZJ_N_FEATURES: usize = 108;
const ZJ_N_INPUTS: usize = 109;
const ZJ_N_CELLS: usize = 36;

struct ZenjpegPicker {
    model: Model,
    features: Vec<AnalysisFeature>,
}

impl ZenjpegPicker {
    fn load(dir: &std::path::Path) -> Option<Self> {
        let order = std::fs::read_to_string(dir.join("feature_order.txt")).ok()?;
        let bytes = std::fs::read(dir.join("picker_zenjpeg_a_v3_f16.bin")).ok()?;
        let by_name: HashMap<&str, AnalysisFeature> = FeatureSet::SUPPORTED
            .iter()
            .map(|f| (f.name(), f))
            .collect();
        let mut features = Vec::with_capacity(ZJ_N_FEATURES);
        for line in order.lines() {
            let Some(col) = line.split('\t').nth(1).map(str::trim) else {
                continue;
            };
            if col.is_empty() {
                continue;
            }
            let name = col.strip_prefix("feat_").unwrap_or(col);
            // A name this build no longer defines is exactly the case where the
            // real picker returns None and the encoder keeps its heuristic.
            features.push(*by_name.get(name)?);
        }
        assert_eq!(
            features.len(),
            ZJ_N_FEATURES,
            "zenjpeg feature_order.txt resolved {} of {ZJ_N_FEATURES} — the bake or this \
             mirror changed; re-read zenjpeg/src/encode/picker.rs before trusting this run",
            features.len()
        );
        // Aligned copy: ZNPR wants 4-byte alignment and `Vec<u8>` gives 1.
        let model = Model::from_bytes(&bytes).ok()?;
        Some(Self { model, features })
    }

    /// Raw predicted outputs (log-bytes per cell) for a feature vector.
    fn scores(&self, res: &AnalysisResults, target: f32) -> Option<Vec<f32>> {
        let mut x = [0.0f32; ZJ_N_INPUTS];
        for (i, &f) in self.features.iter().enumerate() {
            let v = res.get_f32(f).unwrap_or(0.0);
            x[i] = if v.is_finite() { v } else { 0.0 };
        }
        x[ZJ_N_FEATURES] = (target / 100.0).clamp(0.0, 1.0);
        let mut predictor = Predictor::new(&self.model);
        let out = if self.model.has_nontrivial_feature_transforms() {
            predictor.predict_transformed(&x).ok()?
        } else {
            predictor.predict(&x).ok()?
        };
        Some(out[..ZJ_N_CELLS].to_vec())
    }

    /// `Some(cell_index)`; the 36 cells decode as
    /// subsampling{420,422,444} × progressive{f,t} × sharp_yuv{f,t} × effort{0,1,2}.
    fn pick(&self, res: &AnalysisResults, target: f32) -> Option<usize> {
        let mut x = [0.0f32; ZJ_N_INPUTS];
        for (i, &f) in self.features.iter().enumerate() {
            let v = res.get_f32(f).unwrap_or(0.0);
            x[i] = if v.is_finite() { v } else { 0.0 };
        }
        x[ZJ_N_FEATURES] = (target / 100.0).clamp(0.0, 1.0);
        let mut predictor = Predictor::new(&self.model);
        let bounds = self.model.feature_bounds();
        if !bounds.is_empty() && zenpredict::first_out_of_distribution(&x, bounds).is_some() {
            return None;
        }
        let out = if self.model.has_nontrivial_feature_transforms() {
            predictor.predict_transformed(&x).ok()?
        } else {
            predictor.predict(&x).ok()?
        };
        let allow = [true; ZJ_N_CELLS];
        let mask = AllowedMask::new(&allow);
        argmin_masked_in_range(out, (0, ZJ_N_CELLS), &mask, ScoreTransform::Exp, None)
    }
}

fn zj_cell_label(cell: usize) -> String {
    let sub = ["420", "422", "444"][(cell / 12) % 3];
    let prog = if (cell / 6) % 2 == 1 { "prog" } else { "base" };
    let sharp = if (cell / 3) % 2 == 1 {
        "sharp"
    } else {
        "plain"
    };
    let effort = ["fast", "balanced", "max"][cell % 3];
    format!("{sub}/{prog}/{sharp}/{effort}")
}

// ---------------------------------------------------------------- zenwebp
//
// MIRROR of `zenwebp/src/encoder/analysis/classifier.rs::decide_bucket_from_diag`
// (line 664) as of 2026-08-30.
//
// The real function tests seven rules, but four of the fields it reads
// (`screen_content`, `text_likelihood`, `natural_likelihood`, `line_art_score`)
// are hardcoded to 0.0 by its only populated constructor, `diag_from_lookup`
// (classifier.rs:474-482), because those features were culled from zenanalyze.
// Every rule that depends on one of them is therefore dead, and the live rule
// set is the three below. Six of the ten features the classifier requests never
// reach the decision at all.
fn zenwebp_label(res: &AnalysisResults, side: u32) -> &'static str {
    if side <= 128 {
        return "Icon"; // classifier.rs:79 size carve-out
    }
    let get = |name: &str| -> f32 {
        AnalysisFeature::from_name(name)
            .and_then(|f| res.get_f32(f))
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    };
    if get("skin_tone_fraction") >= 0.15 && get("edge_slope_stdev") < 35.0 {
        return "Photo";
    }
    if get("flat_color_block_ratio") >= 0.50 && get("distinct_color_bins") < 4096.0 {
        return "Drawing";
    }
    "Photo"
}

// ---------------------------------------------------------------- main

fn env_list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("ZENANALYZE_CORPUS_DIR").unwrap_or_else(|_| "../codec-corpus".into()),
    );
    let zj_dir = PathBuf::from(
        std::env::var("ZENJPEG_PICKER_DIR")
            .unwrap_or_else(|_| "../zenjpeg/zenjpeg/src/encode/picker_data".into()),
    );
    let out_path =
        std::env::var("AB_OUT").unwrap_or_else(|_| "benchmarks/budget_decision_ab.tsv".to_string());
    let want_sides = env_list("AB_SIDES", &ALL_SIDES.map(|s| s.to_string()).join(","));
    let want_classes = env_list(
        "AB_CLASSES",
        "photo,photohard,screen,mixed,screenwide,lineart",
    );
    let crops: usize = std::env::var("AB_CROPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let bridge = !std::env::var("AB_NO_BRIDGE").is_ok_and(|v| v == "1");
    if bridge {
        eprintln!(
            "NOTE: re-qualifying chroma_subsample_dct_loss to the routers' baked \
             @48f0f976 so the meta-picker answers at all (this build emits @fabc9776). \
             Applied to BOTH arms. AB_NO_BRIDGE=1 to disable."
        );
    }
    let zj = ZenjpegPicker::load(&zj_dir);
    if zj.is_none() {
        eprintln!("WARNING: no zenjpeg bake at {zj_dir:?} — skipping that consumer");
    }

    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
    writeln!(
        out,
        "consumer\tclass\tside\tcrop\ttarget\tdecision_default\tdecision_full\tchanged\tregret"
    )
    .unwrap();

    let mut all: Vec<Class> = classes(&corpus);
    all.extend(extra_classes(&corpus));
    let avail = [
        zenpicker::CodecFamily::Jpeg,
        zenpicker::CodecFamily::Webp,
        zenpicker::CodecFamily::Jxl,
        zenpicker::CodecFamily::Avif,
        zenpicker::CodecFamily::Png,
    ];
    let est = [0u32; zenpicker::CodecFamily::COUNT];

    for class in all
        .iter()
        .filter(|c| want_classes.iter().any(|w| w == c.name))
    {
        for side in ALL_SIDES
            .iter()
            .copied()
            .filter(|s| want_sides.iter().any(|w| w == &s.to_string()))
        {
            for crop in 0..crops {
                eprintln!("{} {side}² crop{crop}", class.name);
                let (buf, _) = build(side, class, crop);
                let r_def = analyze(&buf, side, DEFAULT_PIXEL_BUDGET, DEFAULT_HF_MAX_BLOCKS);
                let r_full = analyze(&buf, side, usize::MAX, HF_UNCAPPED);

                // --- zenwebp content label (size-only input besides features)
                let (a, b) = (zenwebp_label(&r_def, side), zenwebp_label(&r_full, side));
                writeln!(
                    out,
                    "zenwebp_classifier\t{}\t{side}\t{crop}\t-\t{a}\t{b}\t{}\t-",
                    class.name,
                    u8::from(a != b)
                )
                .unwrap();

                // --- cross-codec meta-picker: REAL shipped entry point
                // Lend borrowed offers, per the idiom documented in src/offer.rs.
                let owned_def = offer_cells(&r_def, bridge);
                let owned_full = offer_cells(&r_full, bridge);
                let borrowed_def: Vec<_> =
                    owned_def.iter().map(OwnedFeatureResult::as_ref).collect();
                let borrowed_full: Vec<_> =
                    owned_full.iter().map(OwnedFeatureResult::as_ref).collect();
                let prov = Provenance::new(zenanalyze::analyzer_version());
                let off_def = Offer::new(&borrowed_def, prov);
                let off_full = Offer::new(&borrowed_full, prov);
                for t in targets() {
                    let route = |o: &Offer<'_>| {
                        zenpicker::default_route(
                            o,
                            zenpicker::QualityTarget::Zq(t),
                            &avail,
                            EncodeMode::QueuedBalanced,
                            None,
                            &est,
                        )
                        .ok()
                        .flatten()
                        .map(|d| format!("{:?}", d.family()))
                        .unwrap_or_else(|| "none".into())
                    };
                    let (a, b) = (route(&off_def), route(&off_full));
                    writeln!(
                        out,
                        "meta_picker\t{}\t{side}\t{crop}\t{t}\t{a}\t{b}\t{}\t-",
                        class.name,
                        u8::from(a != b)
                    )
                    .unwrap();
                }

                // --- zenjpeg picker (mirror)
                if let Some(zj) = &zj {
                    for t in targets() {
                        let (Some(ca), Some(cb)) = (zj.pick(&r_def, t), zj.pick(&r_full, t)) else {
                            continue;
                        };
                        // REGRET: what the default-budget pick costs, priced by
                        // the better-informed (fully sampled) feature vector.
                        // The bake's outputs are predicted log-bytes, so
                        // exp(Δlog) − 1 is the model's own estimate of the
                        // fractional byte penalty for having picked `ca`
                        // instead of `cb`. Zero when the picks agree. This is
                        // the model's opinion, NOT a measured encode — but it
                        // is the exact quantity the picker optimises, so a
                        // regret of ~0 means the flip is churn by the picker's
                        // own objective.
                        let regret = match zj.scores(&r_full, t) {
                            Some(s) => (s[ca] - s[cb]).exp() - 1.0,
                            None => f32::NAN,
                        };
                        writeln!(
                            out,
                            "zenjpeg_picker\t{}\t{side}\t{crop}\t{t}\t{}\t{}\t{}\t{regret:.6e}",
                            class.name,
                            zj_cell_label(ca),
                            zj_cell_label(cb),
                            u8::from(ca != cb)
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
    out.flush().unwrap();
    eprintln!("wrote {out_path}");
}
