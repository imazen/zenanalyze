//! Shared machinery for the cross-version graph-runtime gate.
//!
//! `zp_old` is zenpredict at zenanalyze `e6c72f99` (the last commit before
//! ZNPR v4); `zp_new` is this tree. [`run_old`] and [`run_new`] are one
//! template instantiated against each crate, so both sides execute the
//! same calls on the same inputs and are compared by bit pattern.

use std::fmt::Write as _;

/// Everything observable about one model on one side.
#[derive(Debug, PartialEq, Eq)]
pub struct Observed {
    /// `Ok(summary)` or `Err(Debug of PredictError)`.
    pub load: Result<String, String>,
    /// Per vector: output bit patterns, or the error's Debug string.
    pub predict: Vec<Result<Vec<u32>, String>>,
    pub predict_transformed: Vec<Result<Vec<u32>, String>>,
    pub predict_with_specs: Vec<Result<String, String>>,
    pub argmin: Vec<Result<Option<usize>, String>>,
}

/// Feature vectors for one model: `features` are `n_inputs` long (the
/// `predict` path); `raw` are `caller_input_width` long (the
/// `predict_transformed` path). Both include deliberately wrong-length
/// vectors so error paths are compared too.
pub struct Vectors {
    pub features: Vec<Vec<f32>>,
    pub raw: Vec<Vec<f32>>,
}

/// Deterministic xorshift64*, so a failing vector is reproducible from
/// its seed.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

const SPECIALS: [f32; 12] = [
    0.0,
    -0.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE,
    1.0e-40, // subnormal
    f32::MAX,
    -f32::MAX,
    1.0e30,
    -1.0e30,
    1.0,
];

/// Build the vector set. Mix: realistic (`mean ± 3·scale`), wide
/// dynamic range, rows sprinkled with special values, all-zero, and
/// wrong-length vectors.
pub fn make_vectors(
    n_inputs: usize,
    caller_width: usize,
    mean: &[f32],
    scale: &[f32],
    n: usize,
    seed: u64,
) -> Vectors {
    let mut rng = Rng(seed | 1);
    let gen_row = |rng: &mut Rng, k: usize, len: usize, scaled: bool| -> Vec<f32> {
        (0..len)
            .map(|i| match k % 4 {
                0 | 1 if scaled && i < mean.len() => {
                    let s = if scale[i] == 0.0 { 1.0 } else { scale[i] };
                    mean[i] + s * rng.range(-3.0, 3.0)
                }
                0 | 1 => rng.range(-3.0, 3.0),
                2 => {
                    let m = rng.range(-1.0, 1.0);
                    m * 2f32.powi(rng.below(60) as i32 - 30)
                }
                _ => {
                    if rng.below(8) == 0 {
                        SPECIALS[rng.below(SPECIALS.len())]
                    } else {
                        rng.range(-5.0, 5.0)
                    }
                }
            })
            .collect()
    };
    let mut features = Vec::with_capacity(n + 3);
    let mut raw = Vec::with_capacity(n + 3);
    for k in 0..n {
        features.push(gen_row(&mut rng, k, n_inputs, true));
        raw.push(gen_row(&mut rng, k, caller_width, caller_width == n_inputs));
    }
    features.push(vec![0.0; n_inputs]);
    raw.push(vec![0.0; caller_width]);
    features.push(vec![0.5; n_inputs + 1]);
    raw.push(vec![0.5; caller_width + 1]);
    if n_inputs > 1 {
        features.push(vec![0.5; n_inputs - 1]);
    }
    if caller_width > 1 {
        raw.push(vec![0.5; caller_width - 1]);
    }
    Vectors { features, raw }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

macro_rules! side {
    ($name:ident, $zp:ident) => {
        /// Load, describe and run every vector on one side.
        pub fn $name(bytes: &[u8], vectors: &Vectors) -> Observed {
            use $zp::{AllowedMask, Model, OutputValue, Predictor, ScoreTransform, WeightStorage};
            let model = match Model::from_bytes(bytes) {
                Ok(m) => m,
                Err(e) => {
                    return Observed {
                        load: Err(format!("{e:?}")),
                        predict: vec![],
                        predict_transformed: vec![],
                        predict_with_specs: vec![],
                        argmin: vec![],
                    };
                }
            };
            let mut s = String::new();
            let _ = write!(
                s,
                "in={} out={} layers={} scratch={} caller={} expanded={} schema={:#x} transforms={:?} specs={} sparse={} bounds={}",
                model.n_inputs(),
                model.n_outputs(),
                model.n_layers(),
                model.scratch_len(),
                model.caller_input_width(),
                model.expanded_input_dim(),
                model.schema_hash(),
                model.feature_transforms(),
                model.output_specs().len(),
                model.sparse_overrides().len(),
                model.feature_bounds().len(),
            );
            for l in model.layers() {
                let (dtype, wbits): (&str, Vec<u32>) = match &l.weights {
                    WeightStorage::F32(w) => ("f32", w.iter().map(|x| x.to_bits()).collect()),
                    WeightStorage::F16(w) => ("f16", w.iter().map(|&x| x as u32).collect()),
                    WeightStorage::I8 { weights, scales } => (
                        "i8",
                        weights
                            .iter()
                            .map(|&x| x as u8 as u32)
                            .chain(scales.iter().map(|x| x.to_bits()))
                            .collect(),
                    ),
                };
                let h = wbits
                    .iter()
                    .fold(0xcbf2_9ce4_8422_2325u64, |h, &v| (h ^ v as u64).wrapping_mul(0x100_0000_01b3));
                let hb = l
                    .biases
                    .iter()
                    .fold(0xcbf2_9ce4_8422_2325u64, |h, &v| {
                        (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3)
                    });
                let _ = write!(
                    s,
                    " [{}x{} {:?} {dtype} w={h:016x} b={hb:016x}]",
                    l.in_dim, l.out_dim, l.activation
                );
            }
            let mut p = Predictor::new(&model);
            let predict = vectors
                .features
                .iter()
                .map(|f| p.predict(f).map(bits).map_err(|e| format!("{e:?}")))
                .collect();
            let predict_transformed = vectors
                .raw
                .iter()
                .map(|f| p.predict_transformed(f).map(bits).map_err(|e| format!("{e:?}")))
                .collect();
            let predict_with_specs = vectors
                .features
                .iter()
                .map(|f| {
                    p.predict_with_specs(f)
                        .map(|vals| {
                            vals.iter()
                                .map(|v| match v {
                                    OutputValue::Override(x) => format!("O{:08x}", x.to_bits()),
                                    OutputValue::Default => "D".to_string(),
                                    #[allow(unreachable_patterns)]
                                    other => format!("{other:?}"),
                                })
                                .collect::<Vec<_>>()
                                .join(",")
                        })
                        .map_err(|e| format!("{e:?}"))
                })
                .collect();
            let allow = vec![true; model.n_outputs()];
            let mask = AllowedMask::new(&allow);
            let argmin = vectors
                .features
                .iter()
                .map(|f| {
                    p.argmin_masked(f, &mask, ScoreTransform::Identity, None)
                        .map_err(|e| format!("{e:?}"))
                })
                .collect();
            Observed {
                load: Ok(s),
                predict,
                predict_transformed,
                predict_with_specs,
                argmin,
            }
        }
    };
}

side!(run_old, zp_old);
side!(run_new, zp_new);

/// Vectors for `bytes`, sized from whichever side loads it.
pub fn vectors_for(bytes: &[u8], n: usize, seed: u64) -> Vectors {
    if let Ok(m) = zp_old::Model::from_bytes(bytes) {
        return make_vectors(
            m.n_inputs(),
            m.caller_input_width(),
            m.scaler_mean(),
            m.scaler_scale(),
            n,
            seed,
        );
    }
    if let Ok(m) = zp_new::Model::from_bytes(bytes) {
        return make_vectors(
            m.n_inputs(),
            m.caller_input_width(),
            m.scaler_mean(),
            m.scaler_scale(),
            n,
            seed,
        );
    }
    make_vectors(4, 4, &[], &[], 4, seed)
}

/// Outcome of comparing one model across versions.
pub struct Compared {
    pub loaded: bool,
    pub vectors: usize,
    pub values: usize,
    /// Human-readable first mismatches (empty = identical).
    pub mismatches: Vec<String>,
}

pub fn compare(bytes: &[u8], n: usize, seed: u64) -> Compared {
    let v = vectors_for(bytes, n, seed);
    let a = run_old(bytes, &v);
    let b = run_new(bytes, &v);
    let mut mismatches = Vec::new();
    if a.load != b.load {
        mismatches.push(format!("load: old={:?}\n      new={:?}", a.load, b.load));
    }
    let mut values = 0;
    let mut check = |what: &str, i: usize, x: String, y: String| {
        if x != y && mismatches.len() < 8 {
            mismatches.push(format!("{what}[{i}]: old={x}\n      new={y}"));
        }
    };
    for (i, (x, y)) in a.predict.iter().zip(&b.predict).enumerate() {
        if let Ok(o) = x {
            values += o.len();
        }
        check("predict", i, format!("{x:?}"), format!("{y:?}"));
    }
    for (i, (x, y)) in a
        .predict_transformed
        .iter()
        .zip(&b.predict_transformed)
        .enumerate()
    {
        if let Ok(o) = x {
            values += o.len();
        }
        check("predict_transformed", i, format!("{x:?}"), format!("{y:?}"));
    }
    for (i, (x, y)) in a
        .predict_with_specs
        .iter()
        .zip(&b.predict_with_specs)
        .enumerate()
    {
        check("predict_with_specs", i, format!("{x:?}"), format!("{y:?}"));
    }
    for (i, (x, y)) in a.argmin.iter().zip(&b.argmin).enumerate() {
        check("argmin", i, format!("{x:?}"), format!("{y:?}"));
    }
    if a.predict.len() != b.predict.len()
        || a.predict_transformed.len() != b.predict_transformed.len()
    {
        mismatches.push("vector count differs".into());
    }
    Compared {
        loaded: a.load.is_ok(),
        vectors: v.features.len() + v.raw.len(),
        values,
        mismatches,
    }
}

/// A two-layer chain baked with the old composer (`dtype` 0=f32 1=f16
/// 2=i8), LeakyReLU hidden, Identity out — the zenpredict-bake `predict`
/// bench shapes.
pub fn bench_shape(n_in: usize, n_hidden: usize, n_out: usize, dtype: u8, seed: u64) -> Vec<u8> {
    use zp_old::{Activation, WeightDtype};
    use zpb_old::{BakeLayer, BakeRequest, bake};
    let mut rng = Rng(seed | 1);
    let dt = match dtype {
        0 => WeightDtype::F32,
        1 => WeightDtype::F16,
        _ => WeightDtype::I8,
    };
    let mean: Vec<f32> = (0..n_in).map(|_| rng.range(-1.0, 1.0)).collect();
    let scale: Vec<f32> = (0..n_in).map(|_| rng.range(0.5, 1.5)).collect();
    let w0: Vec<f32> = (0..n_in * n_hidden).map(|_| rng.range(-0.3, 0.3)).collect();
    let w1: Vec<f32> = (0..n_hidden * n_out)
        .map(|_| rng.range(-0.3, 0.3))
        .collect();
    let (b0, b1) = (vec![0.0; n_hidden], vec![0.0; n_out]);
    let layers = [
        BakeLayer {
            in_dim: n_in,
            out_dim: n_hidden,
            activation: Activation::LeakyRelu,
            dtype: dt,
            weights: &w0,
            biases: &b0,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_out,
            activation: Activation::Identity,
            dtype: dt,
            weights: &w1,
            biases: &b1,
        },
    ];
    bake(&BakeRequest::new(0, 0, &mean, &scale, &layers)).expect("bake shape")
}
