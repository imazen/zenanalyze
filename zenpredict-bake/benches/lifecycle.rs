//! Synthetic 984->128->64->1 lifecycle workloads; no image/label I/O.
mod common;
use zenbench::black_box;
use zenpredict::{Model, Predictor};
zenbench::main!(|suite| {
    for &name in common::CASES {
        let f = common::fixture(name);
        let bytes: &'static [u8] = Box::leak(f.bytes.into_boxed_slice());
        let source: &'static [f32] = Box::leak(f.source.into_boxed_slice());
        let model: &'static Model = Box::leak(Box::new(Model::from_bytes(bytes).unwrap()));
        suite.group(name, |g| {
            g.bench("load", move |b| {
                b.iter(|| black_box(Model::from_bytes(black_box(bytes)).unwrap()))
            });
            g.bench("construct", move |b| {
                b.iter(|| black_box(Predictor::new(black_box(model)).unwrap()))
            });
            g.bench("predict", move |b| {
                let mut predictor = Predictor::new(model).unwrap();
                b.iter(move || {
                    black_box(predictor.predict_from_source(black_box(source)).unwrap()[0])
                })
            });
        });
    }
});
