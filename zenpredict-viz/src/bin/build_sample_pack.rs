//! Build `web/sample_pack.json` — a sidecar of ~15-20 real
//! (image, codec, q)-style feature vectors drawn from the canonical
//! training parquets at `/mnt/v/zen/zensim-training/canonical-2026-05-21/`.
//!
//! The forward-pass panel's "Load sample" dropdown uses this to
//! prepopulate the feature textarea with meaningful inputs spanning the
//! full quality range (B0..B9), so the viz reports a believable score
//! waterfall instead of the prior `sin(0.1·i)·5.0` synthetic probe.
//!
//! The sidecar is COMMITTED to the repo (under `web/sample_pack.json`)
//! and `build.sh` does NOT regenerate it — the canonical parquets live
//! on the author's NAS (/mnt/v) and CI runners don't have them.
//! Regenerate manually with:
//!
//! ```sh
//! cargo run --release -p zenpredict-viz \
//!     --features sample-pack \
//!     --bin build_sample_pack
//! ```
//!
//! Schema-locked to 372 features (the canonical width); 228- and
//! 300-input bakes show "samples not available" in the UI. Adding
//! per-schema sample pools is a follow-up — the trainer's older bake
//! widths aren't in the canonical-2026-05-21 set.

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::array::{Array, Float64Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;

const CANONICAL_ROOT: &str = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train";
const N_FEATURES: usize = 372;

/// One (image, codec, q)-style sample row from a canonical parquet.
struct CandidateRow {
    /// Stable identifier (e.g. `safesyn_b9_00b13be94a4867dd_1022x818_a42`).
    id: String,
    /// Human-readable label for the dropdown.
    label: String,
    /// Predicted "expected score" using the corpus's most-trustworthy
    /// 0..100-shaped column. Reported as guidance only — the actual
    /// score depends on the loaded bake.
    expected_score: f64,
    /// Which column drove `expected_score`.
    expected_score_source: &'static str,
    /// Corpus tag: `safesyn`, `konjnd-dense`, `kadid`, `tid`, ...
    source_corpus: &'static str,
    /// Original source image basename (informational; usually does not
    /// resolve to a viewable file in the viz).
    ref_basename: String,
    /// Length-N_FEATURES feature vector.
    features: Vec<f64>,
}

#[derive(Serialize)]
struct SamplePack {
    /// Feature schema (number of inputs expected by the bake).
    /// Locked to 372 for the MVP.
    schema: usize,
    /// Generated UTC date (YYYY-MM-DD).
    generated_at: String,
    /// Source parquets used to extract these samples.
    source_parquets: Vec<String>,
    /// Pre-computed score for the identity-image baseline (all-zero
    /// features). Stored for the UI hint; the actual identity score is
    /// bake-dependent.
    notes: String,
    samples: Vec<SampleRecord>,
}

#[derive(Serialize)]
struct SampleRecord {
    id: String,
    label: String,
    expected_score: f64,
    expected_score_source: &'static str,
    source_corpus: &'static str,
    ref_basename: String,
    n_features: usize,
    features: Vec<f64>,
}

/// Required columns for sample extraction.
const COL_REF: &str = "ref_basename";
const COL_MIX: &str = "mix_cv40_iw60";
const COL_SSIM2: &str = "ssim2_gpu";
const COL_PJND: &str = "pjnd_target";

fn main() {
    let manifest_dir = PathBuf::from(env_required("CARGO_MANIFEST_DIR"));
    let out_path = manifest_dir.join("web/sample_pack.json");

    let canonical_root = Path::new(CANONICAL_ROOT);
    if !canonical_root.exists() {
        eprintln!(
            "error: canonical training root not found at {}\n\
             the sample_pack sidecar is built from the author's NAS;\n\
             commit the existing web/sample_pack.json instead, or fix the path.",
            canonical_root.display()
        );
        std::process::exit(2);
    }

    let safesyn_path = canonical_root.join("safesyn.parquet");
    let konjnd_path = canonical_root.join("konjnd-dense.parquet");
    let kadid_path = canonical_root.join("kadid.parquet");
    let tid_path = canonical_root.join("tid.parquet");

    let mut all_candidates = Vec::<CandidateRow>::new();

    // Each (corpus_tag, score-band predicate, target_count, source col) request:
    // we pull from safesyn (the broadest synth set) for most bands, and from
    // konjnd-dense for PJND-anchored samples. KADID and TID provide a
    // line-art / authentic-distortion variety pair.

    // Safesyn — bulk samples across ssim2 bands.
    let safesyn_targets: &[(&str, f64, f64, usize)] = &[
        // (band_label, ssim2_lo, ssim2_hi, count)
        ("highq_near_lossless", 96.0, 100.0, 1),
        ("highq_photo", 88.0, 95.0, 2),
        ("midhi_q", 75.0, 87.0, 2),
        ("mid_q", 60.0, 72.0, 2),
        ("midlo_q", 45.0, 58.0, 2),
        ("lo_q", 28.0, 42.0, 2),
        ("verylo_q", 14.0, 25.0, 1),
        ("pathological", 0.0, 12.0, 2),
    ];
    let safesyn_rows =
        read_corpus_rows(&safesyn_path, true).expect("failed to read safesyn parquet");
    pick_band_samples(
        &safesyn_rows,
        safesyn_targets,
        "safesyn",
        &mut all_candidates,
    );

    // KonJND-dense — explicitly anchor the PJND boundary (~58-68
    // on `human_score` which mirrors the dense-mix target). The dense
    // set has `pjnd_target` non-null for every row, which lets us bias
    // toward the actual JND threshold.
    let konjnd_rows = read_corpus_rows(&konjnd_path, true).expect("failed to read konjnd parquet");
    pick_pjnd_anchors(&konjnd_rows, "konjnd-dense", &mut all_candidates, 2);

    // KADID — synthetic-distortion corpus where ssim2 clusters bimodally
    // (~125 rows in [93,94] from the heaviest distortion levels, ~10k
    // rows in [99.5, 100] from milder distortion levels). Pick one
    // from each cluster so the dropdown exposes both regimes.
    let kadid_rows = read_corpus_rows(&kadid_path, true).expect("failed to read kadid parquet");
    pick_band_samples(
        &kadid_rows,
        &[
            ("kadid_mild_distortion", 99.5, 100.0, 1),
            ("kadid_heavy_distortion", 93.0, 95.0, 1),
        ],
        "kadid",
        &mut all_candidates,
    );

    // TID — 1 sample spanning the broader 60..78 ssim2 band.
    let tid_rows = read_corpus_rows(&tid_path, true).expect("failed to read tid parquet");
    pick_band_samples(
        &tid_rows,
        &[("tid_mid", 64.0, 74.0, 1)],
        "tid",
        &mut all_candidates,
    );

    // Synthetic identity-image baseline — all features zero. Real
    // bakes shipped against scaled inputs may not literally read 100,
    // but this exposes the "what does the network do at the origin?"
    // case so users can see the implicit bias.
    all_candidates.push(CandidateRow {
        id: "identity_zero_features".into(),
        label: "identity (all-zero features) — scaler-origin probe".into(),
        expected_score: f64::NAN, // bake-dependent; the UI will skip rendering the delta
        expected_score_source: "synthetic zero vector",
        source_corpus: "synthetic",
        ref_basename: "(zero-vector probe)".into(),
        features: vec![0.0; N_FEATURES],
    });

    // Deduplicate any near-identical picks by ref_basename + band-bucket.
    // (Safesyn often has many distortions on the same ref; we already
    // limit one-per-ref via `pick_band_samples`, but a defensive dedup
    // here doesn't hurt.)
    let mut seen_ids: HashSet<String> = HashSet::new();
    all_candidates.retain(|c| seen_ids.insert(c.id.clone()));

    println!("→ assembled {} samples", all_candidates.len());
    for c in &all_candidates {
        let exp = if c.expected_score.is_nan() {
            "(bake-dependent)".to_string()
        } else {
            format!("{:.2}", c.expected_score)
        };
        println!(
            "    {:<48} expected={:<8} corpus={:<12} src={}",
            c.label, exp, c.source_corpus, c.ref_basename
        );
    }

    let pack = SamplePack {
        schema: N_FEATURES,
        generated_at: utc_today(),
        source_parquets: vec![
            "canonical-2026-05-21/train/safesyn.parquet".into(),
            "canonical-2026-05-21/train/konjnd-dense.parquet".into(),
            "canonical-2026-05-21/train/kadid.parquet".into(),
            "canonical-2026-05-21/train/tid.parquet".into(),
        ],
        notes: "Schema-locked to 372 features. \
                `expected_score` is `mix_cv40_iw60` from the source row \
                (or `human_score` for konjnd-dense). Actual score \
                depends on the loaded bake's calibration."
            .into(),
        samples: all_candidates
            .into_iter()
            .map(|c| SampleRecord {
                id: c.id,
                label: c.label,
                expected_score: c.expected_score,
                expected_score_source: c.expected_score_source,
                source_corpus: c.source_corpus,
                ref_basename: c.ref_basename,
                n_features: c.features.len(),
                features: c.features,
            })
            .collect(),
    };

    let json = serde_json::to_string_pretty(&pack).expect("failed to serialize sample_pack json");
    std::fs::write(&out_path, json).expect("failed to write sample_pack.json");
    println!(
        "✓ wrote {} ({} samples, schema={})",
        out_path.display(),
        pack.samples.len(),
        pack.schema
    );

    // Diagnostic: score every sample through v_tuner_v11 + v_balanced_v3
    // so the developer can sanity-check the predicted range matches the
    // sample's stated band. The shipping bakes live in zensim/weights/
    // relative to the repo workspace. Skip silently if not present.
    let zensim_weights = manifest_dir.join("../../zensim/zensim/weights");
    for bake_name in &["v_tuner_v11_2026-05-24.bin", "v_balanced_v3_2026-05-20.bin"] {
        let bake_path = zensim_weights.join(bake_name);
        if !bake_path.exists() {
            eprintln!(
                "  (skip diagnostic: {} not found at {})",
                bake_name,
                bake_path.display()
            );
            continue;
        }
        let Ok(bake_bytes) = std::fs::read(&bake_path) else {
            eprintln!("  (skip: failed to read {})", bake_path.display());
            continue;
        };
        let Ok(model) = zenpredict::Model::from_bytes(&bake_bytes) else {
            eprintln!("  (skip: failed to parse {})", bake_path.display());
            continue;
        };
        if model.n_inputs() != N_FEATURES {
            eprintln!(
                "  (skip {}: n_inputs={} != {})",
                bake_name,
                model.n_inputs(),
                N_FEATURES
            );
            continue;
        }
        let mut predictor = zenpredict::Predictor::new(&model);
        println!(
            "\n=== diagnostic scoring on {} ({} bytes) ===",
            bake_name,
            bake_bytes.len()
        );
        let mut scores: Vec<f32> = Vec::new();
        for sample in &pack.samples {
            let feats: Vec<f32> = sample.features.iter().map(|v| *v as f32).collect();
            match predictor.predict_transformed(&feats) {
                Ok(out) => {
                    let v = out[0];
                    let exp_str = if sample.expected_score.is_nan() {
                        "    (none)".to_string()
                    } else {
                        format!("{:>8.3}", sample.expected_score)
                    };
                    println!(
                        "  predicted={:>8.3}  expected={}  {}",
                        v, exp_str, sample.label
                    );
                    scores.push(v);
                }
                Err(e) => {
                    println!("  predict failed for {}: {:?}", sample.id, e);
                }
            }
        }
        if !scores.is_empty() {
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo = *scores.first().unwrap();
            let hi = *scores.last().unwrap();
            let med = scores[scores.len() / 2];
            println!(
                "  range [{:.2}, {:.2}]  median {:.2}  spread {:.2}",
                lo,
                hi,
                med,
                hi - lo
            );
        }
    }
}

/// In-memory view of a parquet file's relevant columns.
struct CorpusRows {
    ref_basenames: Vec<String>,
    mix_cv40_iw60: Vec<Option<f64>>,
    ssim2_gpu: Vec<Option<f64>>,
    human_score: Vec<Option<f64>>,
    pjnd_target: Vec<Option<f64>>,
    features: Vec<Vec<f64>>,
}

/// Load all rows from a canonical parquet, projecting only the columns
/// we need + `f0..f371`. Returns a column-of-rows view; the caller
/// picks samples by indexing into matching rows.
fn read_corpus_rows(path: &Path, has_pjnd: bool) -> Result<CorpusRows, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();
    let arrow_fields = schema.fields();

    // Build a name -> arrow-index map so projection is robust to column
    // reordering between corpora.
    let name_to_idx =
        |needle: &str| -> Option<usize> { arrow_fields.iter().position(|f| f.name() == needle) };

    let ref_idx = name_to_idx(COL_REF).ok_or_else(|| format!("{path:?}: missing {COL_REF}"))?;
    let mix_idx = name_to_idx(COL_MIX);
    let ssim2_idx = name_to_idx(COL_SSIM2);
    let human_idx = name_to_idx("human_score");
    let pjnd_idx = if has_pjnd {
        name_to_idx(COL_PJND)
    } else {
        None
    };

    let f0_idx = name_to_idx("f0").ok_or_else(|| format!("{path:?}: missing f0"))?;
    // Sanity-check we have 372 consecutive feature cols.
    for i in 0..N_FEATURES {
        let need = format!("f{i}");
        let got = arrow_fields[f0_idx + i].name();
        if got != &need {
            return Err(format!(
                "{path:?}: expected {need} at index {}, got {got}",
                f0_idx + i
            ));
        }
    }

    // Project just the columns we care about (cheaper than reading the
    // full 394-col rowgroup).
    let mut projection_idxs = vec![ref_idx];
    for opt in [mix_idx, ssim2_idx, human_idx, pjnd_idx].iter().flatten() {
        projection_idxs.push(*opt);
    }
    for i in 0..N_FEATURES {
        projection_idxs.push(f0_idx + i);
    }

    // Map arrow column indices to parquet leaf indices for the
    // projection mask. The Arrow / Parquet schemas line up 1:1 for
    // flat primitive types, which is what all our cols are.
    let leaf_idxs: Vec<usize> = projection_idxs.to_vec();
    let projection =
        parquet::arrow::ProjectionMask::leaves(&parquet_schema, leaf_idxs.iter().copied());

    let reader = builder
        .with_projection(projection)
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{path:?}: build reader: {e}"))?;

    let mut ref_basenames = Vec::<String>::new();
    let mut mix = Vec::<Option<f64>>::new();
    let mut ssim2 = Vec::<Option<f64>>::new();
    let mut human = Vec::<Option<f64>>::new();
    let mut pjnd = Vec::<Option<f64>>::new();
    let mut features: Vec<Vec<f64>> = Vec::new();

    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: read batch: {e}"))?;
        let cols = batch.columns();
        // After projection, columns appear in the order we requested.
        // First col is always ref_basename, then up to 4 optional float
        // cols, then 372 feature cols. We reverse-map by name from
        // batch.schema() to stay robust if parquet emits the cols in
        // a different order than requested.
        let bs = batch.schema();
        let bs_fields = bs.fields();
        let get_col =
            |name: &str| -> Option<usize> { bs_fields.iter().position(|f| f.name() == name) };
        let ref_b = get_col(COL_REF).expect("ref column dropped by projection");
        let mix_b = get_col(COL_MIX);
        let ssim2_b = get_col(COL_SSIM2);
        let human_b = get_col("human_score");
        let pjnd_b = get_col(COL_PJND);

        let ref_arr = cols[ref_b]
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| format!("{path:?}: ref column not a string"))?;
        let mix_arr = mix_b.and_then(|i| cols[i].as_any().downcast_ref::<Float64Array>());
        let ssim2_arr = ssim2_b.and_then(|i| cols[i].as_any().downcast_ref::<Float64Array>());
        let human_arr = human_b.and_then(|i| cols[i].as_any().downcast_ref::<Float64Array>());
        let pjnd_arr = pjnd_b.and_then(|i| cols[i].as_any().downcast_ref::<Float64Array>());

        // Per-row feature builders: precompute the f0..f371 batch column
        // indices once.
        let mut feat_batch_idxs = Vec::<usize>::with_capacity(N_FEATURES);
        for i in 0..N_FEATURES {
            let name = format!("f{i}");
            let idx = get_col(&name).ok_or_else(|| format!("{path:?}: missing {name} in batch"))?;
            feat_batch_idxs.push(idx);
        }
        let feat_arrs: Vec<&Float64Array> = feat_batch_idxs
            .iter()
            .map(|&i| {
                cols[i]
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("feature col not f64")
            })
            .collect();

        for row in 0..batch.num_rows() {
            ref_basenames.push(ref_arr.value(row).to_string());
            mix.push(read_opt(&mix_arr, row));
            ssim2.push(read_opt(&ssim2_arr, row));
            human.push(read_opt(&human_arr, row));
            pjnd.push(read_opt(&pjnd_arr, row));
            let mut feats = Vec::with_capacity(N_FEATURES);
            for arr in &feat_arrs {
                if arr.is_null(row) {
                    feats.push(0.0);
                } else {
                    feats.push(arr.value(row));
                }
            }
            features.push(feats);
        }
    }

    Ok(CorpusRows {
        ref_basenames,
        mix_cv40_iw60: mix,
        ssim2_gpu: ssim2,
        human_score: human,
        pjnd_target: pjnd,
        features,
    })
}

fn read_opt(arr: &Option<&Float64Array>, row: usize) -> Option<f64> {
    arr.and_then(|a| {
        if a.is_null(row) {
            None
        } else {
            Some(a.value(row))
        }
    })
}

/// Pick `count` samples per band from the source corpus, where each
/// band is `(label, ssim2_lo, ssim2_hi, count)`. Picks deterministic
/// representative rows (the median-ssim2 row in each band) and avoids
/// re-picking the same `ref_basename` twice within the same call.
fn pick_band_samples(
    rows: &CorpusRows,
    bands: &[(&str, f64, f64, usize)],
    corpus_tag: &'static str,
    out: &mut Vec<CandidateRow>,
) {
    let mut used_refs: HashSet<String> = HashSet::new();

    for (band_label, lo, hi, count) in bands.iter().copied() {
        // Gather candidates that fall in the band by ssim2 (or
        // mix_cv40_iw60 if ssim2 is missing — konjnd-dense never
        // reaches here because that path uses pick_pjnd_anchors).
        let mut indices: Vec<usize> = (0..rows.ref_basenames.len())
            .filter(|&i| {
                let s = rows.ssim2_gpu[i].or(rows.mix_cv40_iw60[i]);
                s.map(|v| v >= lo && v <= hi).unwrap_or(false)
            })
            .collect();
        if indices.is_empty() {
            eprintln!(
                "  ! no rows in band {band_label} [{lo:.1}, {hi:.1}] for {corpus_tag}; skipping"
            );
            continue;
        }
        // Sort by ssim2 for deterministic spread; pick `count` evenly
        // distributed indices across the band.
        indices.sort_by(|&a, &b| {
            let av = rows.ssim2_gpu[a].or(rows.mix_cv40_iw60[a]).unwrap_or(0.0);
            let bv = rows.ssim2_gpu[b].or(rows.mix_cv40_iw60[b]).unwrap_or(0.0);
            av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_avail = indices.len();
        let stride = ((n_avail as f64) / (count as f64 + 1.0)).max(1.0);
        let mut picked = 0;
        let mut probe = 0usize;
        while picked < count && probe < n_avail {
            let slot = ((picked as f64 + 1.0) * stride) as usize + probe;
            if slot >= n_avail {
                break;
            }
            let idx = indices[slot.min(n_avail - 1)];
            let refn = &rows.ref_basenames[idx];
            if used_refs.contains(refn) {
                probe += 1;
                continue;
            }
            used_refs.insert(refn.clone());

            let ssim2 = rows.ssim2_gpu[idx];
            let mix = rows.mix_cv40_iw60[idx];
            let human = rows.human_score[idx];

            // Expected-score preference: for safesyn, mix_cv40_iw60 is
            // the canonical training target. For kadid/tid, it's also
            // present but compressed; ssim2 gives a wider scale that
            // matches what zensim ships will predict.
            let (exp, exp_src) = match corpus_tag {
                "safesyn" => mix
                    .map(|v| (v, "mix_cv40_iw60"))
                    .or_else(|| ssim2.map(|v| (v, "ssim2_gpu")))
                    .unwrap_or((f64::NAN, "unknown")),
                "kadid" | "tid" => ssim2
                    .map(|v| (v, "ssim2_gpu"))
                    .or_else(|| mix.map(|v| (v, "mix_cv40_iw60")))
                    .unwrap_or((f64::NAN, "unknown")),
                _ => human
                    .map(|v| (v, "human_score"))
                    .or_else(|| ssim2.map(|v| (v, "ssim2_gpu")))
                    .unwrap_or((f64::NAN, "unknown")),
            };

            let label = format_label(corpus_tag, band_label, exp, exp_src, refn);
            let id = format!(
                "{corpus_tag}_{band_label}_{refn}",
                refn = refn
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_')
                    .collect::<String>()
            );
            out.push(CandidateRow {
                id,
                label,
                expected_score: exp,
                expected_score_source: exp_src,
                source_corpus: corpus_tag,
                ref_basename: refn.clone(),
                features: rows.features[idx].clone(),
            });
            picked += 1;
            probe += 1;
        }
        if picked < count {
            eprintln!(
                "  ! only picked {picked}/{count} in band {band_label} (ran out of \
                 unique refs in [{lo:.1}, {hi:.1}])"
            );
        }
    }
}

/// Pick `count` samples from konjnd-dense near the PJND threshold
/// (~63 ± 5 per the CID22 paper, anchoring "visually lossless"). The
/// dense set's `human_score` carries the dense-mix target; rows where
/// `human_score` lies in the Near-PJND band [58, 68] are the canonical
/// JND-boundary samples.
fn pick_pjnd_anchors(
    rows: &CorpusRows,
    corpus_tag: &'static str,
    out: &mut Vec<CandidateRow>,
    count: usize,
) {
    let mut indices: Vec<usize> = (0..rows.ref_basenames.len())
        .filter(|&i| {
            rows.human_score[i]
                .map(|v| (58.0..=68.0).contains(&v))
                .unwrap_or(false)
        })
        .collect();
    if indices.is_empty() {
        eprintln!("  ! no Near-PJND rows in {corpus_tag}");
        return;
    }
    indices.sort_by(|&a, &b| {
        let av = rows.human_score[a].unwrap_or(0.0);
        let bv = rows.human_score[b].unwrap_or(0.0);
        av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = indices.len();
    let stride = (n / (count + 1)).max(1);
    let mut used_refs: HashSet<String> = HashSet::new();
    let mut picked = 0;
    let mut slot = stride;
    while picked < count && slot < n {
        let idx = indices[slot];
        slot += stride;
        let refn = &rows.ref_basenames[idx];
        if !used_refs.insert(refn.clone()) {
            continue;
        }
        let exp = rows.human_score[idx].unwrap_or(f64::NAN);
        let pjnd = rows.pjnd_target[idx];
        let pjnd_note = match pjnd {
            Some(v) => format!(" (PJND target {v:.2})"),
            None => String::new(),
        };
        let label = format!("Near-PJND boundary ≈ {exp:.1}{pjnd_note} — {corpus_tag} {refn}");
        let id = format!(
            "{corpus_tag}_pjnd_{refn_safe}",
            refn_safe = refn
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
        );
        out.push(CandidateRow {
            id,
            label,
            expected_score: exp,
            expected_score_source: "human_score (dense-mix)",
            source_corpus: corpus_tag,
            ref_basename: refn.clone(),
            features: rows.features[idx].clone(),
        });
        picked += 1;
    }
}

fn format_label(corpus: &str, band: &str, expected: f64, src: &str, refn: &str) -> String {
    let band_short = band.replace('_', " ");
    if expected.is_nan() {
        format!("{corpus} {band_short} — {refn}")
    } else {
        format!("{corpus} {band_short} (≈{expected:.1} via {src}) — {refn}")
    }
}

fn utc_today() -> String {
    // Avoid pulling in chrono / time — use the platform `date` if
    // available; otherwise fall back to the build date macro.
    match std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%d"])
        .output()
    {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".into(),
    }
}

fn env_required(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| {
        eprintln!("required env var {key} is not set");
        std::process::exit(2);
    })
}
