//! End-to-end demo of the INERT cell-layout path: real zenanalyze feature
//! values, keyed by their qualified `name@hex8` identities, routed through the
//! bake's own contract to a `family × mode` pick.
//!
//! This is the shape a real consumer has — features named by the analyzer, not
//! by slot. The `metapicker_v1` bake declares its inputs positionally
//! (`feat_0..feat_60`), so the demo resolves names through the recovered slot
//! table (`benchmarks/metapicker_v1_feature_slots_2026-08-30.tsv`, compiled in)
//! and then hands `CellContract::build_input` a by-name source. Nothing here
//! touches `default_route` or the shipped routers.
//!
//!   cargo run --release -p zenpicker --example cell_pick_demo -- \
//!       /mnt/v/output/zensim/metapicker-2026-08-30/metapicker_v1.bin \
//!       /mnt/v/output/clean-picker-corpus-2026-06-26/clean_features.tsv \
//!       82            # target zq
//!       [row]         # 1-based data row of the TSV (default 1)

use std::collections::HashMap;

use zenpicker::CellPicker;

/// slot -> qualified zenanalyze name, recovered from the builder's own rule.
const SLOT_TABLE: &str =
    include_str!("../../benchmarks/metapicker_v1_feature_slots_2026-08-30.tsv");

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!(
            "usage: cell_pick_demo <bake.bin> <clean_features.tsv> <target_zq 0..100> [row=1]"
        );
        std::process::exit(2);
    }
    let target_zq: f32 = a[3].parse().expect("target_zq is a number");
    let want_row: usize = a.get(4).map_or(1, |s| s.parse().expect("row is a number"));

    let bytes = std::fs::read(&a[1]).expect("read bake");
    let picker = CellPicker::from_znpr_bytes(&bytes).expect("bake satisfies the cell contract");
    let contract = picker.contract();

    // slot name (as the bake declares it) -> qualified zenanalyze name
    let slot_to_name: HashMap<&str, &str> = SLOT_TABLE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .skip(1)
        .filter_map(|l| {
            let mut f = l.split('\t');
            let _slot = f.next()?;
            Some((f.next()?.trim(), f.next()?.trim()))
        })
        .collect();

    // One rendition's features, keyed by the analyzer's qualified names.
    let tsv = std::fs::read_to_string(&a[2]).expect("read features tsv");
    let mut lines = tsv.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let row: Vec<&str> = lines
        .nth(want_row - 1)
        .unwrap_or_else(|| panic!("row {want_row} not in the tsv"))
        .split('\t')
        .collect();
    let by_name: HashMap<&str, f32> = header
        .iter()
        .zip(&row)
        .filter_map(|(h, v)| v.parse::<f32>().ok().map(|x| (*h, x)))
        .collect();
    println!("rendition: {}", row.first().copied().unwrap_or("?"));

    // THE contract mapping: each declared slot resolved to its analyzer name,
    // read exactly once. A name the caller cannot supply is a loud error.
    let input = contract
        .build_input(target_zq / 100.0, |slot| {
            by_name.get(slot_to_name.get(slot)?).copied()
        })
        .expect("every contract feature supplied");

    // Route among the families this caller can emit. A real consumer would
    // intersect its own format mask with the contract's, e.g.
    //   contract.families().intersect(AllowedFamilies::from_allowed(my_formats))
    let allowed = contract.families();
    let pred = picker
        .predict_cells(&input, &allowed, None)
        .expect("forward pass");

    println!("target_zq {target_zq}  ->  predicted bytes_log per cell:");
    for (cell, score) in contract.cells().iter().zip(pred.scores()) {
        let mark = if pred.pick().map(|p| p.label()) == Some(cell.label()) {
            " <- PICK"
        } else {
            ""
        };
        println!("  {:<18} {:>9.4}{mark}", cell.label(), score);
    }
    match pred.pick() {
        Some(c) => println!(
            "pick: {} (family {:?}, mode {:?})",
            c.label(),
            c.family(),
            c.mode()
        ),
        None => println!("pick: none (every cell masked out)"),
    }
}
