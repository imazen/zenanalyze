//! Build `web/feature_catalog.json` — semantic metadata for every
//! feature ID across the zensim 228/300/372 schemas and zenanalyze's
//! per-image feature set.
//!
//! Source of truth:
//!   - zensim: per-block feature names + descriptions from the doc
//!     comments in `zensim/src/metric.rs` (around the constants
//!     `FEATURES_PER_CHANNEL_BASIC` / `_WITH_PEAKS` / `_EXTENDED` /
//!     `_IW`). The names are hand-mirrored here because the trainer
//!     itself emits these names; the schema is locked at the wire
//!     format level.
//!   - zenanalyze: parsed from the `features_table!` macro in
//!     `zenanalyze/src/feature.rs`. The macro is small enough to
//!     extract via line-by-line text scan; we look for
//!     `<Variant> = <id> : <ty> => <field>` rows and grab the
//!     preceding /// doc comments.
//!
//! No training-stats block in MVP — wiring up a parquet reader is a
//! follow-up. The JSON schema includes a `null` `training_stats`
//! field so the JS side knows the slot exists.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;

const ZENANALYZE_REPO_REL: &str = "../../zenanalyze";
const ZENSIM_REPO_REL: &str = "../../../zensim";

#[derive(Serialize)]
struct Catalog {
    schema_version: u32,
    /// Generated UTC timestamp (informational).
    generated_at: String,
    /// Zensim per-bake feature schemas — keyed by total feature count.
    zensim_schemas: BTreeMap<String, ZensimSchema>,
    /// zenanalyze source-image feature catalog — keyed by numeric id
    /// (string for JSON-friendly stable ordering).
    zenanalyze_features: BTreeMap<String, ZenanalyzeFeature>,
}

#[derive(Serialize)]
struct ZensimSchema {
    n_features: usize,
    n_scales: usize,
    n_channels: usize,
    channels: Vec<String>,
    /// Each block: (tag, names_per_channel_per_scale).
    blocks: Vec<Block>,
    /// Per-index entry. Keyed `"f0"`, `"f1"`, …, `"f<n-1>"`.
    features: BTreeMap<String, FeatureEntry>,
}

#[derive(Serialize)]
struct Block {
    tag: String,
    description: String,
    /// Names within this block (per channel per scale).
    names: Vec<String>,
}

#[derive(Serialize)]
struct FeatureEntry {
    label: String,
    block: String,
    scale: usize,
    channel: String,
    feature_name: String,
    math_summary: String,
    source_ref: String,
    training_stats: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ZenanalyzeFeature {
    id: u32,
    variant: String,
    field_name: String,
    ty: String,
    summary: String,
    source_ref: String,
}

fn main() {
    let manifest_dir = env_required("CARGO_MANIFEST_DIR");
    let manifest_dir = PathBuf::from(manifest_dir);
    let zenanalyze_root = manifest_dir.join(ZENANALYZE_REPO_REL);
    let zensim_root = manifest_dir.join(ZENSIM_REPO_REL);

    let mut catalog = Catalog {
        schema_version: 1,
        generated_at: utc_now_iso(),
        zensim_schemas: BTreeMap::new(),
        zenanalyze_features: BTreeMap::new(),
    };

    let zensim_metric_path = zensim_root.join("zensim/src/metric.rs");
    let zensim_metric_ref = relative_source_ref(&zensim_metric_path);

    catalog
        .zensim_schemas
        .insert("228".into(), build_zensim_schema(228, &zensim_metric_ref));
    catalog
        .zensim_schemas
        .insert("300".into(), build_zensim_schema(300, &zensim_metric_ref));
    catalog
        .zensim_schemas
        .insert("372".into(), build_zensim_schema(372, &zensim_metric_ref));

    let zenanalyze_feature_path = zenanalyze_root.join("src/feature.rs");
    let zenanalyze_feature_ref = relative_source_ref(&zenanalyze_feature_path);
    match std::fs::read_to_string(&zenanalyze_feature_path) {
        Ok(src) => {
            for entry in parse_features_table(&src, &zenanalyze_feature_ref) {
                catalog
                    .zenanalyze_features
                    .insert(format!("{}", entry.id), entry);
            }
        }
        Err(e) => {
            eprintln!(
                "warn: could not read {}: {e}",
                zenanalyze_feature_path.display()
            );
        }
    }

    let out_path = manifest_dir.join("web/feature_catalog.json");
    let json = serde_json::to_string_pretty(&catalog).expect("serialize catalog");
    std::fs::write(&out_path, &json).expect("write feature_catalog.json");
    eprintln!(
        "✓ wrote {} ({} zensim schemas, {} zenanalyze features)",
        out_path.display(),
        catalog.zensim_schemas.len(),
        catalog.zenanalyze_features.len(),
    );
}

fn env_required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        panic!("env var {name} not set — run via `cargo run` so manifest dir is exported")
    })
}

fn utc_now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = dur.as_secs() as i64;
    // Minimal RFC3339 without bringing in chrono.
    let (year, month, day, hour, minute, second) = epoch_to_ymdhms(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn epoch_to_ymdhms(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    // Civil-from-days algorithm by Howard Hinnant (public domain).
    let z = secs.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = (y + if m <= 2 { 1 } else { 0 }) as i32;
    let sec_of_day = secs.rem_euclid(86_400) as u32;
    let hour = sec_of_day / 3600;
    let minute = (sec_of_day % 3600) / 60;
    let second = sec_of_day % 60;
    (year, m as u32, d as u32, hour, minute, second)
}

fn relative_source_ref(p: &Path) -> String {
    p.canonicalize()
        .map(|c| {
            // Try to render relative to the user's home so the entry
            // stays readable even when /home/<user>/work/ varies.
            if let Some(home) = std::env::var_os("HOME") {
                let home = PathBuf::from(home);
                if let Ok(rel) = c.strip_prefix(&home) {
                    return format!("~/{}", rel.display());
                }
            }
            c.display().to_string()
        })
        .unwrap_or_else(|_| p.display().to_string())
}

/// Build a zensim feature schema for a given total feature count.
fn build_zensim_schema(n_features: usize, source_ref: &str) -> ZensimSchema {
    let n_scales = 4usize;
    let channels = vec!["Y".to_string(), "Cb".to_string(), "Cr".to_string()];

    let basic = BASIC_FEATURES;
    let peaks = PEAK_FEATURES;
    let masked = MASKED_FEATURES;
    let iw = IW_FEATURES;

    let mut blocks: Vec<Block> = Vec::new();
    blocks.push(Block {
        tag: "basic".into(),
        description: "13 always-on per-(channel, scale) statistics over SSIM error, edge artifact, edge detail-lost, MSE, and HF energy/magnitude ratios.".into(),
        names: basic.iter().map(|(n, _)| n.to_string()).collect(),
    });
    blocks.push(Block {
        tag: "peaks".into(),
        description: "6 per-(channel, scale) peak/L8 statistics: max + (Σd⁸/N)^(1/8) of SSIM/art/det maps. Surfaces tail behavior the means/L2 average over.".into(),
        names: peaks.iter().map(|(n, _)| n.to_string()).collect(),
    });
    if n_features >= 300 {
        blocks.push(Block {
            tag: "masked".into(),
            description: "6 per-(channel, scale) flatness-mask-weighted statistics. masked_ssim_mean / 4th / 2nd, masked_art_4th, masked_det_4th, masked_mse. Down-weights texture regions where small errors are less visible.".into(),
            names: masked.iter().map(|(n, _)| n.to_string()).collect(),
        });
    }
    if n_features >= 372 {
        blocks.push(Block {
            tag: "iw".into(),
            description: "6 per-(channel, scale) Wang & Li 2011 information-content-weighted (IW-SSIM) statistics. Polarity-flipped vs `masked`: texture-rich regions get MORE weight.".into(),
            names: iw.iter().map(|(n, _)| n.to_string()).collect(),
        });
    }

    // Per-feature emission.
    let per_scale = 3 * sum_block_widths(&blocks);
    let n_block_widths: Vec<usize> = blocks.iter().map(|b| b.names.len()).collect();
    let mut features: BTreeMap<String, FeatureEntry> = BTreeMap::new();
    for idx in 0..n_features {
        let scale = idx / per_scale;
        let within_scale = idx % per_scale;
        let per_channel_total = sum_block_widths(&blocks);
        let channel_idx = within_scale / per_channel_total;
        let within_channel = within_scale % per_channel_total;
        let (block_idx, feature_in_block) = locate_block(&n_block_widths, within_channel);
        let block_tag = blocks[block_idx].tag.clone();
        let feature_name = blocks[block_idx].names[feature_in_block].clone();
        let math_summary = lookup_math_summary(&block_tag, &feature_name);
        let channel = channels[channel_idx].clone();
        let label = format!("s{scale}.{channel}.{feature_name}");
        features.insert(
            format!("f{idx}"),
            FeatureEntry {
                label,
                block: block_tag,
                scale,
                channel,
                feature_name,
                math_summary,
                source_ref: source_ref.to_string(),
                training_stats: None,
            },
        );
    }

    ZensimSchema {
        n_features,
        n_scales,
        n_channels: 3,
        channels,
        blocks,
        features,
    }
}

fn sum_block_widths(blocks: &[Block]) -> usize {
    blocks.iter().map(|b| b.names.len()).sum()
}

fn locate_block(widths: &[usize], within_channel: usize) -> (usize, usize) {
    let mut offset = 0usize;
    for (idx, &w) in widths.iter().enumerate() {
        if within_channel < offset + w {
            return (idx, within_channel - offset);
        }
        offset += w;
    }
    (widths.len() - 1, widths[widths.len() - 1] - 1)
}

// =============== Zensim feature names & descriptions ==============

/// (name, math_summary). Names mirror the table in
/// `zensim/src/metric.rs` (the `FEATURES_PER_CHANNEL_BASIC` doc block).
const BASIC_FEATURES: &[(&str, &str)] = &[
    ("ssim_mean", "mean(SSIM error map)"),
    (
        "ssim_4th",
        "(Σ d⁴ / N)^(1/4) over SSIM error map (L4 pooling)",
    ),
    (
        "ssim_2nd",
        "(Σ d² / N)^(1/2) over SSIM error map (L2 / RMSE pooling)",
    ),
    ("art_mean", "mean of edge-artifact map (ringing)"),
    ("art_4th", "L4 pool of edge-artifact map"),
    ("art_2nd", "L2 pool of edge-artifact map"),
    ("det_mean", "mean of edge-detail-lost map (blur)"),
    ("det_4th", "L4 pool of edge-detail-lost map"),
    ("det_2nd", "L2 pool of edge-detail-lost map"),
    ("mse", "mean((src - dst)²) per channel"),
    (
        "hf_energy_loss",
        "max(0, 1 - Σ(dst-μ)²/Σ(src-μ)²) — local detail energy lost",
    ),
    (
        "hf_mag_loss",
        "max(0, 1 - Σ|dst-μ|/Σ|src-μ|) — L1 detail magnitude lost",
    ),
    (
        "hf_energy_gain",
        "max(0, Σ(dst-μ)²/Σ(src-μ)² - 1) — added local energy (ringing)",
    ),
];

const PEAK_FEATURES: &[(&str, &str)] = &[
    ("ssim_max", "max(per-pixel SSIM error)"),
    ("art_max", "max(per-pixel edge-artifact)"),
    ("det_max", "max(per-pixel edge-detail-lost)"),
    (
        "ssim_l8",
        "(Σ d⁸ / N)^(1/8) over SSIM error map (heavy-tail pooling)",
    ),
    ("art_l8", "L8 pool of edge-artifact map"),
    ("det_l8", "L8 pool of edge-detail-lost map"),
];

const MASKED_FEATURES: &[(&str, &str)] = &[
    (
        "masked_ssim_mean",
        "mean(SSIM × flatness_mask) — small errors in flat regions",
    ),
    ("masked_ssim_4th", "L4 pool of masked SSIM error"),
    ("masked_ssim_2nd", "L2 pool of masked SSIM error"),
    ("masked_art_4th", "L4 pool of masked edge-artifact"),
    ("masked_det_4th", "L4 pool of masked edge-detail-lost"),
    ("masked_mse", "mean((src-dst)² × flatness_mask)"),
];

const IW_FEATURES: &[(&str, &str)] = &[
    (
        "iw_ssim_mean",
        "mean(SSIM × iw_weight) — texture-EMPHASISED SSIM (Wang & Li 2011)",
    ),
    ("iw_ssim_4th", "L4 pool of IW SSIM"),
    ("iw_ssim_2nd", "L2 pool of IW SSIM"),
    ("iw_art_4th", "L4 pool of IW edge-artifact"),
    ("iw_det_4th", "L4 pool of IW edge-detail-lost"),
    ("iw_mse", "mean((src-dst)² × iw_weight)"),
];

fn lookup_math_summary(block_tag: &str, feature_name: &str) -> String {
    let table: &[(&str, &str)] = match block_tag {
        "basic" => BASIC_FEATURES,
        "peaks" => PEAK_FEATURES,
        "masked" => MASKED_FEATURES,
        "iw" => IW_FEATURES,
        _ => &[],
    };
    table
        .iter()
        .find_map(|(n, m)| {
            if *n == feature_name {
                Some((*m).to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| feature_name.to_string())
}

// =============== zenanalyze features_table! parser ================

/// Parse the `features_table!` macro body in zenanalyze/src/feature.rs.
/// The grammar (informal):
///   /// summary line A
///   /// summary line B
///   #[cfg(feature = "experimental")]
///   <Variant> = <id> : <ty> => <field_name>,
/// We collect contiguous `///` doc comments above a row and tie them
/// to the row's id.
fn parse_features_table(src: &str, source_ref: &str) -> Vec<ZenanalyzeFeature> {
    let mut out: Vec<ZenanalyzeFeature> = Vec::new();
    let mut in_table = false;
    let mut pending_docs: Vec<String> = Vec::new();
    for raw in src.lines() {
        let line = raw.trim();
        if !in_table {
            if line.starts_with("features_table!") {
                in_table = true;
            }
            continue;
        }
        if line == "}" {
            // End of macro body. (We're not robust to multiple `}` —
            // the macro body is a single brace group.)
            break;
        }
        if let Some(rest) = line.strip_prefix("///") {
            pending_docs.push(rest.trim().to_string());
            continue;
        }
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with("#[")
            || line.starts_with("@decl[")
            || line.starts_with(")")
        {
            // Skip non-doc attributes / blank lines without dropping the
            // accumulated docs (the row's docs may sit above a #[cfg]).
            continue;
        }
        // Try to parse a row: `Variant = ID : Ty => field,`
        if let Some(entry) = parse_row(line, &pending_docs, source_ref) {
            out.push(entry);
        }
        pending_docs.clear();
    }
    out
}

fn parse_row(line: &str, docs: &[String], source_ref: &str) -> Option<ZenanalyzeFeature> {
    // Strip trailing comma.
    let line = line.trim_end_matches(',').trim();
    let (lhs, rhs) = line.split_once("=>")?;
    let lhs = lhs.trim();
    let field = rhs.trim().trim_end_matches(',').trim();
    let (variant_part, type_part) = lhs.split_once(':')?;
    let (variant_id, id_str) = variant_part.trim().split_once('=')?;
    let variant = variant_id.trim().to_string();
    let id: u32 = id_str.trim().parse().ok()?;
    let ty = type_part.trim().to_string();
    let summary = if docs.is_empty() {
        format!("(undocumented feature: {variant})")
    } else {
        docs.join(" ")
    };
    Some(ZenanalyzeFeature {
        id,
        variant,
        field_name: field.to_string(),
        ty,
        summary,
        source_ref: source_ref.to_string(),
    })
}
