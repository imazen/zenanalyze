//! Measure a baked router's family-accuracy on a dumped test set — loads the REAL ZNPR
//! `.bin` (so f32 vs i8 is a ground-truth comparison, not a numpy replica). Each test line is
//! `<label>\t<v0>\t<v1>...` (raw inputs; the model's scaler runs inside predict). The pick is
//! the masked argmin over the branch family indices (same as `RouteDecision::resolve`).
//!
//!   cargo run --release -p zenpicker --features api --example score_router -- \
//!       router_lossy_i8.bin router_lossy_test.tsv 0,1,2,3
use std::fs;
use zenpredict::{Model, Predictor};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: score_router <model.bin> <test.tsv> <branch_idx_csv e.g. 0,1,2,3>");
        std::process::exit(2);
    }
    let model_bytes = Aligned(fs::read(&a[1]).unwrap());
    let model = Model::from_bytes(&model_bytes.0).unwrap();
    let mut p = Predictor::new(&model);
    let branch: Vec<usize> = a[3].split(',').map(|s| s.trim().parse().unwrap()).collect();

    let txt = fs::read_to_string(&a[2]).unwrap();
    let (mut correct, mut total) = (0usize, 0usize);
    for line in txt.lines() {
        let mut it = line.split('\t');
        let label: usize = it.next().unwrap().parse().unwrap();
        let x: Vec<f32> = it.map(|s| s.parse().unwrap()).collect();
        let out = p.predict(&x).unwrap();
        // masked argmin over the branch family indices (lower = better)
        let pick = branch
            .iter()
            .copied()
            .min_by(|&i, &j| out[i].partial_cmp(&out[j]).unwrap())
            .unwrap();
        if pick == label {
            correct += 1;
        }
        total += 1;
    }
    println!(
        "{}: {} rows, family-acc = {:.2}%",
        a[1].rsplit('/').next().unwrap(),
        total,
        100.0 * correct as f64 / total as f64
    );
}
