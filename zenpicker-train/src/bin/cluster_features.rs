//! `cluster_features` — Rust/linfa port of
//! `zenpicker-train/scripts/cluster_sources.py`.
//!
//! Cluster source images on their zenanalyze content features
//! (`feat_*` columns) and pick `k` centroid-nearest representatives —
//! the zensim/CLAUDE.md "Dense sampling for trained models"
//! stratification rule (pick representative sources via k-means on a
//! feature-space embedding, choosing the centroid-nearest member of
//! each cluster rather than random sampling).
//!
//! This is a behaviour-for-behaviour port of `cluster_sources.py`,
//! swapping numpy/sklearn for `ndarray` + `linfa-clustering`
//! (pure-Rust, no system BLAS). The pipeline is identical:
//!
//! 1. Read the input (TSV or parquet); discover `feat_*` columns,
//!    sorted numerically (`feat_2 < feat_10`).
//! 2. Build an `n_rows × n_feat` `f64` matrix; empty / unparseable
//!    cells become `NaN`, then each `NaN` is filled with that
//!    column's median (non-finite median → `0.0`).
//! 3. Drop zero-variance columns (population std ≤ `1e-9`).
//! 4. Z-score standardize each surviving column (population std).
//! 5. Run linfa `KMeans` with `k = min(k, n_rows)` clusters,
//!    `n_runs = 10`, `tolerance = 1e-4` (sklearn-default-ish), seeded
//!    by `--seed`.
//! 6. For each non-empty cluster, pick the member nearest the centroid
//!    (Euclidean in standardized space) and emit its `image_path`.
//! 7. Write the newline path list (`--out-list`) and the JSON sidecar
//!    (`--out-json`) in the same shape as the Python tool.
//!
//! Both TSV and parquet inputs are wired (parquet via the same
//! `feat_*` schema-discovery pattern used by `parquet_input.rs`).
//!
//! NOTE: k-means cluster *labels* are initialization-dependent, so the
//! integer cluster ids and their order need not match sklearn's for the
//! same seed — but the *partition* (which images group together) and
//! the centroid-nearest representatives are the algorithmic contract
//! this port preserves. The RNG is `rand_xoshiro::Xoshiro256Plus`
//! (linfa's default), not numpy's MT19937, so byte-identical picks vs
//! the Python tool are not expected; equivalent picks on well-separated
//! data are.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::ExitCode;

use arrow::array::{Array, Float32Array, Float64Array, Int64Array, StringArray};
use linfa::DatasetBase;
use linfa::traits::{Fit, Predict};
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand_xoshiro::Xoshiro256Plus;

/// Parsed CLI arguments.
struct Args {
    features: String,
    k: usize,
    out_list: String,
    out_json: String,
    seed: u64,
}

const USAGE: &str = "\
cluster_features — k-means stratified source picker (linfa port of cluster_sources.py)

USAGE:
    cluster_features --features <TSV|parquet> --k <N> \\
        --out-list <path> --out-json <path> [--seed <N>]

ARGS:
    --features <path>   extract_features_for_picker TSV, or a parquet with
                        an `image_path` column + `feat_*` feature columns.
    --k <N>             number of clusters / representatives (default 8).
    --out-list <path>   newline-separated chosen image paths.
    --out-json <path>   cluster-assignment JSON sidecar.
    --seed <N>          RNG seed (default 0).
";

fn parse_args() -> Result<Args, String> {
    let mut features: Option<String> = None;
    let mut k: usize = 8;
    let mut out_list: Option<String> = None;
    let mut out_json: Option<String> = None;
    let mut seed: u64 = 0;

    let args_vec: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args_vec.len() {
        let a = args_vec[i].as_str();
        let mut take = |flag: &str| -> Result<String, String> {
            let v = args_vec
                .get(i + 1)
                .cloned()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            i += 1;
            Ok(v)
        };
        match a {
            "--features" => features = Some(take("--features")?),
            "--k" => {
                k = take("--k")?
                    .parse()
                    .map_err(|e| format!("--k not an integer: {e}"))?;
            }
            "--out-list" => out_list = Some(take("--out-list")?),
            "--out-json" => out_json = Some(take("--out-json")?),
            "--seed" => {
                seed = take("--seed")?
                    .parse()
                    .map_err(|e| format!("--seed not an integer: {e}"))?;
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unexpected argument: {other}\n\n{USAGE}")),
        }
        i += 1;
    }

    Ok(Args {
        features: features.ok_or("missing required --features")?,
        k,
        out_list: out_list.ok_or("missing required --out-list")?,
        out_json: out_json.ok_or("missing required --out-json")?,
        seed,
    })
}

/// Raw input: per-row `image_path` plus a row-major `n_rows × n_feat`
/// matrix of `f64` (NaN for empty / unparseable cells), with the
/// feature-column names in matrix-column order.
struct RawInput {
    paths: Vec<String>,
    /// `n_rows × n_feat` row-major; NaN = missing.
    matrix: Array2<f64>,
    feat_names: Vec<String>,
}

/// Stable numeric ordering of `feat_N` so `feat_2 < feat_10` (matches
/// `parquet_input.rs`). Names without a numeric suffix sort last by
/// `u64::MAX`, then lexically.
fn sort_feat_names(names: &mut [String]) {
    names.sort_by(|a, b| {
        let na = a
            .strip_prefix("feat_")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(u64::MAX);
        let nb = b
            .strip_prefix("feat_")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(u64::MAX);
        na.cmp(&nb).then_with(|| a.cmp(b))
    });
}

/// Read the TSV produced by `extract_features_for_picker`: a header
/// row, then one tab-separated row per image. Replicates the Python
/// reader: empty / non-floatable `feat_*` cells become NaN.
fn read_tsv(path: &Path) -> Result<RawInput, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut header_line = String::new();
    if reader
        .read_line(&mut header_line)
        .map_err(|e| e.to_string())?
        == 0
    {
        return Err("empty TSV (no header)".into());
    }
    let header: Vec<&str> = header_line.trim_end_matches(['\n', '\r']).split('\t').collect();

    let path_col = header
        .iter()
        .position(|h| *h == "image_path")
        .ok_or("no `image_path` column in TSV header")?;

    // Discover feat_* columns: (matrix-column index after sort) keyed
    // by original header index.
    let mut feat_names: Vec<String> = header
        .iter()
        .filter(|h| h.starts_with("feat_"))
        .map(|s| s.to_string())
        .collect();
    if feat_names.is_empty() {
        return Err("no feat_* columns found in TSV header".into());
    }
    sort_feat_names(&mut feat_names);
    // Map each feature (in final matrix-column order) to its header index.
    let feat_header_idx: Vec<usize> = feat_names
        .iter()
        .map(|n| header.iter().position(|h| h == n).expect("feat discovered from header"))
        .collect();

    let mut paths: Vec<String> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.trim_end_matches(['\n', '\r']).split('\t').collect();
        let path = cells.get(path_col).copied().unwrap_or("").to_string();
        paths.push(path);
        for &hi in &feat_header_idx {
            let v = cells.get(hi).copied().unwrap_or("");
            let parsed = if v.is_empty() {
                f64::NAN
            } else {
                v.parse::<f64>().unwrap_or(f64::NAN)
            };
            data.push(parsed);
        }
    }

    let n_rows = paths.len();
    let n_feat = feat_names.len();
    let matrix = Array2::from_shape_vec((n_rows, n_feat), data)
        .map_err(|e| format!("TSV row/column count mismatch: {e}"))?;
    Ok(RawInput {
        paths,
        matrix,
        feat_names,
    })
}

/// Pull a numeric arrow column (Float32/Float64/Int64) into `f64`,
/// nulls → NaN. Returns false if the column is not a supported numeric
/// type.
fn numeric_col_to_f64(arr: &dyn Array, out: &mut Vec<f64>) -> bool {
    if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        for i in 0..a.len() {
            out.push(if a.is_null(i) { f64::NAN } else { a.value(i) as f64 });
        }
        true
    } else if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        for i in 0..a.len() {
            out.push(if a.is_null(i) { f64::NAN } else { a.value(i) });
        }
        true
    } else if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        for i in 0..a.len() {
            out.push(if a.is_null(i) { f64::NAN } else { a.value(i) as f64 });
        }
        true
    } else {
        false
    }
}

/// Read a parquet whose schema has an `image_path` utf8 column plus
/// `feat_*` numeric columns. Reuses the `feat_*` discovery convention
/// from `parquet_input.rs`. All rows are kept (no codec / target
/// filtering — clustering has no target); missing cells become NaN.
fn read_parquet(path: &Path) -> Result<RawInput, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    let schema = builder.schema().clone();

    let mut feat_names: Vec<String> = schema
        .fields()
        .iter()
        .filter(|f| f.name().starts_with("feat_"))
        .map(|f| f.name().clone())
        .collect();
    if feat_names.is_empty() {
        return Err("no feat_* columns found in parquet schema".into());
    }
    sort_feat_names(&mut feat_names);

    let col_idx = |name: &str| schema.fields().iter().position(|f| f.name() == name);
    let image_idx = col_idx("image_path").ok_or("no `image_path` column in parquet schema")?;
    let feat_idx: Vec<usize> = feat_names
        .iter()
        .map(|n| col_idx(n).expect("feat discovered from schema"))
        .collect();

    let reader = builder.build().map_err(|e| e.to_string())?;

    let mut paths: Vec<String> = Vec::new();
    // Per-feature column accumulators (full column reads, assembled
    // row-major at the end — mirrors parquet_input.rs).
    let mut feat_cols: Vec<Vec<f64>> = vec![Vec::new(); feat_idx.len()];

    for batch in reader {
        let batch = batch.map_err(|e| e.to_string())?;
        let img = batch
            .column(image_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("image_path column is not utf8")?;
        for i in 0..img.len() {
            paths.push(if img.is_null(i) {
                String::new()
            } else {
                img.value(i).to_string()
            });
        }
        for (slot, &ci) in feat_idx.iter().enumerate() {
            if !numeric_col_to_f64(batch.column(ci), &mut feat_cols[slot]) {
                return Err(format!("feature column `{}` is not numeric", feat_names[slot]));
            }
        }
    }

    let n_rows = paths.len();
    let n_feat = feat_names.len();
    let mut matrix = Array2::<f64>::from_elem((n_rows, n_feat), f64::NAN);
    for (c, col) in feat_cols.iter().enumerate() {
        if col.len() != n_rows {
            return Err(format!(
                "feature column `{}` has {} values but {} rows",
                feat_names[c],
                col.len(),
                n_rows
            ));
        }
        for (r, &v) in col.iter().enumerate() {
            matrix[[r, c]] = v;
        }
    }

    Ok(RawInput {
        paths,
        matrix,
        feat_names,
    })
}

/// Detect input format by extension; default to TSV.
fn read_input(path: &Path) -> Result<RawInput, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("parquet") | Some("pq") => read_parquet(path),
        _ => read_tsv(path),
    }
}

/// Population median of the finite entries of `col`, or `None` if no
/// finite entry exists. Matches `np.nanmedian` (average of the two
/// middle order statistics for an even count).
fn finite_median(col: &[f64]) -> Option<f64> {
    let mut v: Vec<f64> = col.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite only"));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

/// Population standard deviation (ddof=0), matching numpy's default
/// `ndarray.std(axis=0)` (which is also population). `mean` is passed
/// in to avoid recomputation.
fn population_std(col: &[f64], mean: f64) -> f64 {
    let n = col.len();
    if n == 0 {
        return 0.0;
    }
    let var = col.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    var.sqrt()
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    let raw = read_input(Path::new(&args.features))?;
    let n_rows = raw.paths.len();
    if n_rows == 0 {
        return Err("input has no data rows".into());
    }
    let n_feat = raw.feat_names.len();

    // --- NaN -> column median (np.nanmedian; non-finite median -> 0). ---
    let mut matrix = raw.matrix;
    let col_med: Vec<f64> = (0..n_feat)
        .map(|c| {
            let col: Vec<f64> = matrix.column(c).to_vec();
            finite_median(&col).filter(|m| m.is_finite()).unwrap_or(0.0)
        })
        .collect();
    for r in 0..n_rows {
        for c in 0..n_feat {
            if !matrix[[r, c]].is_finite() {
                matrix[[r, c]] = col_med[c];
            }
        }
    }

    // --- Drop zero-variance columns (population std <= 1e-9). ---
    // Mirrors `keep = std > 1e-9` in cluster_sources.py.
    let mut means = vec![0.0_f64; n_feat];
    let mut stds = vec![0.0_f64; n_feat];
    for c in 0..n_feat {
        let col: Vec<f64> = matrix.column(c).to_vec();
        let mean = col.iter().sum::<f64>() / n_rows as f64;
        means[c] = mean;
        stds[c] = population_std(&col, mean);
    }
    let keep: Vec<bool> = stds.iter().map(|s| *s > 1e-9).collect();
    let kept_names: Vec<String> = raw
        .feat_names
        .iter()
        .zip(keep.iter())
        .filter_map(|(n, k)| if *k { Some(n.clone()) } else { None })
        .collect();
    let n_kept = kept_names.len();
    if n_kept == 0 {
        return Err("all feature columns are zero-variance; nothing to cluster".into());
    }

    // --- Z-score standardize surviving columns (population std). ---
    let mut std_x = Array2::<f64>::zeros((n_rows, n_kept));
    {
        let mut kc = 0usize;
        for c in 0..n_feat {
            if !keep[c] {
                continue;
            }
            let m = means[c];
            let s = stds[c];
            for r in 0..n_rows {
                std_x[[r, kc]] = (matrix[[r, c]] - m) / s;
            }
            kc += 1;
        }
    }

    // --- KMeans with k = min(k, n_rows), seeded. ---
    let k = args.k.min(n_rows).max(1);
    let rng = Xoshiro256Plus::seed_from_u64(args.seed);
    let dataset = DatasetBase::from(std_x.clone());
    // n_runs=10 + tolerance=1e-4 mirror sklearn's n_init=10 default and
    // KMeans's documented tolerance; max_n_iterations left at the linfa
    // default (300, same as sklearn's max_iter default).
    let model = KMeans::params_with_rng(k, rng)
        .n_runs(10)
        .tolerance(1e-4)
        .fit(&dataset)
        .map_err(|e| format!("k-means fit failed: {e}"))?;

    // Predict cluster labels for every row.
    let labels = model.predict(&dataset);
    let centroids: &Array2<f64> = model.centroids();

    // --- For each non-empty cluster, pick the centroid-nearest member. ---
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut chosen: Vec<String> = Vec::new();
    for cl in 0..k {
        // Members of this cluster.
        let members: Vec<usize> = (0..n_rows).filter(|&i| labels[i] == cl).collect();
        if members.is_empty() {
            continue;
        }
        let centroid: Array1<f64> = centroids.index_axis(Axis(0), cl).to_owned();
        let mut best_i = members[0];
        let mut best_d = f64::INFINITY;
        for &m in &members {
            let row = std_x.index_axis(Axis(0), m);
            // Euclidean distance in standardized space.
            let d2: f64 = row
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let d = d2.sqrt();
            if d < best_d {
                best_d = d;
                best_i = m;
            }
        }
        chosen.push(raw.paths[best_i].clone());
        clusters.push(Cluster {
            id: cl,
            size: members.len(),
            rep_path: raw.paths[best_i].clone(),
            rep_dist: best_d,
        });
    }

    // Sort clusters largest-first for the JSON + stderr summary (matches
    // `sorted(clusters, key=lambda c: -c["size"])`). `sort_by_key` is a
    // stable sort, so size ties preserve ascending cluster id (clusters
    // were appended in id order).
    clusters.sort_by_key(|c| std::cmp::Reverse(c.size));

    // --- Write outputs. ---
    {
        let mut f =
            File::create(&args.out_list).map_err(|e| format!("{}: {e}", args.out_list))?;
        for p in &chosen {
            writeln!(f, "{p}").map_err(|e| e.to_string())?;
        }
    }
    {
        let json = build_json(k, n_rows, n_kept, &kept_names, &clusters);
        let mut f =
            File::create(&args.out_json).map_err(|e| format!("{}: {e}", args.out_json))?;
        f.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
    }

    eprintln!(
        "chose {} representatives from {} samples, k={}",
        chosen.len(),
        n_rows,
        k
    );
    for c in &clusters {
        eprintln!("  cluster {:2}  size={:4}  {}", c.id, c.size, c.rep_path);
    }
    Ok(())
}

/// One emitted cluster: its (initialization-dependent) label, member
/// count, and the chosen centroid-nearest representative.
#[derive(Clone)]
struct Cluster {
    id: usize,
    size: usize,
    rep_path: String,
    rep_dist: f64,
}

impl ClusterEntry for Cluster {
    fn id(&self) -> usize {
        self.id
    }
    fn size(&self) -> usize {
        self.size
    }
    fn rep_path(&self) -> &str {
        &self.rep_path
    }
    fn rep_dist(&self) -> f64 {
        self.rep_dist
    }
}

/// Build the JSON sidecar with the same key order / shape as
/// `cluster_sources.py` (`json.dump(..., indent=2)`): top-level keys
/// `k`, `n_samples`, `n_features_used`, `features_used`, `clusters`;
/// each cluster entry `cluster`, `size`, `rep_path`, `rep_dist`.
fn build_json(
    k: usize,
    n_samples: usize,
    n_features_used: usize,
    features_used: &[String],
    clusters: &[impl ClusterEntry],
) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"k\": {k},\n"));
    s.push_str(&format!("  \"n_samples\": {n_samples},\n"));
    s.push_str(&format!("  \"n_features_used\": {n_features_used},\n"));
    s.push_str("  \"features_used\": [");
    if features_used.is_empty() {
        s.push(']');
    } else {
        s.push('\n');
        for (i, f) in features_used.iter().enumerate() {
            s.push_str("    ");
            s.push_str(&json_string(f));
            if i + 1 < features_used.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ]");
    }
    s.push_str(",\n");
    s.push_str("  \"clusters\": [");
    if clusters.is_empty() {
        s.push(']');
    } else {
        s.push('\n');
        for (i, c) in clusters.iter().enumerate() {
            s.push_str("    {\n");
            s.push_str(&format!("      \"cluster\": {},\n", c.id()));
            s.push_str(&format!("      \"size\": {},\n", c.size()));
            s.push_str(&format!("      \"rep_path\": {},\n", json_string(c.rep_path())));
            s.push_str(&format!("      \"rep_dist\": {}\n", json_number(c.rep_dist())));
            s.push_str("    }");
            if i + 1 < clusters.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ]");
    }
    s.push_str("\n}");
    s
}

/// Minimal trait so `build_json` doesn't depend on the local `Cluster`
/// struct definition (which is nested inside `run`).
trait ClusterEntry {
    fn id(&self) -> usize;
    fn size(&self) -> usize;
    fn rep_path(&self) -> &str;
    fn rep_dist(&self) -> f64;
}

/// JSON-escape a string (matches Python's `json.dumps` for the ASCII +
/// common control-character cases; image paths are plain UTF-8).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Render an f64 the way `json.dump` does for a Python float: finite
/// values via the shortest round-trippable repr (Rust's `{}` for f64
/// is shortest round-trip), non-finite as their Python literals.
fn json_number(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        // Ensure a decimal point so it reads as a float (Python floats
        // always serialize with one, e.g. `0.0`).
        let mut s = format!("{x}");
        if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            s.push_str(".0");
        }
        s
    }
}
