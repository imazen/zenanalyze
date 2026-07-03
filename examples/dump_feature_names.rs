//! Dump every stable-id -> canonical name pair this build of zenanalyze
//! supports, one `id\tname` line per row, sorted by id.
//!
//! This is the "named-only" enumeration external training pipelines need to
//! correctly filter a joined training parquet: a parquet built from a
//! zenmetrics sweep can carry BOTH zenanalyze's own named features (feat_variance,
//! feat_edge_density, ...) AND an unrelated feature basis from another crate
//! (e.g. zensim's internal feat_0..feat_371 MLP inputs) under the same `feat_`
//! prefix. Naive `startswith("feat_")` filtering can't tell them apart and
//! silently doubles the picker's input width with a foreign, uncurated feature
//! set. Filtering a candidate column list against this dump's names (stripped
//! of the `feat_` prefix and any `@hex8` qualifier) keeps only genuine
//! zenanalyze features.
//!
//! Run: `cargo run --example dump_feature_names [--features experimental,hdr]`

fn main() {
    let n = zenanalyze::feature_count();
    // feature_count() is the count of SUPPORTED ids in this build, not the
    // max id value — ids can be sparse (retired / cfg-gated), so scan a
    // generous range and only print what resolves.
    let mut printed = 0usize;
    for id in 0..(n as u16).saturating_mul(3).max(512) {
        if let Some(name) = zenanalyze::feature_name(id) {
            if zenanalyze::feature_id_supported(id) {
                println!("{id}\t{name}");
                printed += 1;
            }
        }
    }
    eprintln!("# {printed} supported features (feature_count() reports {n})");
}
