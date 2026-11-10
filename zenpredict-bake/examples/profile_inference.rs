//! Fixed-work heaptrack driver. Timing belongs to the lifecycle zenbench target.
#[path = "../benches/common/mod.rs"]
mod common;
use std::hint::black_box;
use zenpredict::{Model, Predictor};
#[inline(never)]
fn load_loop(bytes: &[u8], n: usize) {
    for _ in 0..n {
        black_box(Model::from_bytes(black_box(bytes)).unwrap());
    }
}
#[inline(never)]
fn construct_loop(model: &Model, n: usize) {
    for _ in 0..n {
        black_box(Predictor::new(black_box(model)));
    }
}
#[inline(never)]
fn predict_loop(predictor: &mut Predictor<'_>, source: &[f32], n: usize) {
    for _ in 0..n {
        black_box(predictor.predict_from_source(black_box(source)).unwrap()[0]);
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: profile_inference CASE load|construct|predict ITERATIONS"
    );
    let f = common::fixture(&args[1]);
    let n = args[3].parse().unwrap();
    let model = Model::from_bytes(&f.bytes).unwrap();
    let mut predictor = Predictor::new(&model);
    match args[2].as_str() {
        "load" => load_loop(&f.bytes, n),
        "construct" => construct_loop(&model, n),
        "predict" => predict_loop(&mut predictor, &f.source, n),
        _ => panic!("unknown phase"),
    }
    println!("{} {} {}", args[1], args[2], n);
}
