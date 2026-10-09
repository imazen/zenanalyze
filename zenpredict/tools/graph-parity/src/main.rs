//! Cross-version gate for the ZNPR v4 graph runtime.
//!
//! ```text
//! zenpredict-graph-parity check <bake.bin>...   # every file: old vs new runtime
//! zenpredict-graph-parity synth [cases]         # random v3 chains: old vs new
//!                                               #   composer bytes AND runtime
//! ```
//!
//! `check` loads each file with the pre-graph zenpredict and with this
//! tree, then compares: load verdict (summary or error), every layer's
//! weights and biases, and — over a few thousand realistic, wide-range,
//! special-value and wrong-length feature vectors — `predict`,
//! `predict_transformed`, `predict_with_specs` and `argmin_masked`, by bit
//! pattern. `synth` generates random v3 requests (dtype mixes, all three
//! v3 activations, tail widths, compression, feature/output permutations,
//! output specs, sparse overrides, feature transforms), bakes each with
//! the old and new composer, requires identical bytes, then runs `check`'s
//! comparison on the result. Exit status 1 on any difference.
//!
//! Vectors per model: `ZPGRAPH_VECTORS` (default 2000).

use std::process::ExitCode;

use zenpredict_graph_parity::{Rng, compare};

fn n_vectors() -> usize {
    std::env::var("ZPGRAPH_VECTORS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000)
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

fn check(paths: &[String]) -> ExitCode {
    let n = n_vectors();
    let mut bad = 0;
    println!("status\tloaded\tvectors\tvalues\tbytes\tfnv64\tpath");
    for path in paths {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                println!("ERROR\t-\t-\t-\t-\t-\t{path}: {e}");
                bad += 1;
                continue;
            }
        };
        let c = compare(&bytes, n, fnv(&bytes));
        let status = if c.mismatches.is_empty() {
            "IDENTICAL"
        } else {
            bad += 1;
            "MISMATCH"
        };
        println!(
            "{status}\t{}\t{}\t{}\t{}\t{:016x}\t{path}",
            c.loaded,
            c.vectors,
            c.values,
            bytes.len(),
            fnv(&bytes)
        );
        for m in &c.mismatches {
            println!("    {m}");
        }
    }
    println!("# files={} mismatched={bad}", paths.len());
    if bad == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

const DIMS: [usize; 14] = [1, 2, 3, 5, 7, 8, 9, 13, 16, 17, 24, 31, 33, 64];

fn perm(rng: &mut Rng, n: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        p.swap(i, rng.below(i + 1));
    }
    p
}

macro_rules! bake_with {
    ($zp:ident, $zpb:ident, $case:expr) => {{
        use $zp::{
            Activation, FeatureBound, MetadataType, OutputSpec, SparseOverride, WeightDtype,
        };
        use $zpb::{BakeLayer, BakeMetadataEntry, BakeRequest};
        let c: &Case = $case;
        let layers: Vec<BakeLayer<'_>> = c
            .layers
            .iter()
            .map(|l| BakeLayer {
                in_dim: l.in_dim,
                out_dim: l.out_dim,
                activation: match l.act {
                    0 => Activation::Identity,
                    1 => Activation::Relu,
                    _ => Activation::LeakyRelu,
                },
                dtype: match l.dtype {
                    0 => WeightDtype::F32,
                    1 => WeightDtype::F16,
                    _ => WeightDtype::I8,
                },
                weights: &l.weights,
                biases: &l.biases,
            })
            .collect();
        let bounds: Vec<FeatureBound> = c
            .bounds
            .iter()
            .map(|&(lo, hi)| FeatureBound::new(lo, hi))
            .collect();
        let specs: Vec<OutputSpec> = c
            .specs
            .iter()
            .map(|&(lo, hi)| {
                let mut s = OutputSpec::passthrough();
                s.bounds = FeatureBound::new(lo, hi);
                s
            })
            .collect();
        let sparse: Vec<SparseOverride> = c
            .sparse
            .iter()
            .map(|&(idx, value)| SparseOverride { idx, value })
            .collect();
        let md: Vec<BakeMetadataEntry<'_>> = c
            .transforms
            .as_ref()
            .map(|t| {
                vec![BakeMetadataEntry {
                    key: "zentrain.feature_transforms",
                    kind: MetadataType::Utf8,
                    value: t.as_bytes(),
                }]
            })
            .unwrap_or_default();
        let mut req = BakeRequest::new(c.schema, 0, &c.mean, &c.scale, &layers);
        req.feature_bounds = &bounds;
        req.output_specs = &specs;
        req.sparse_overrides = &sparse;
        req.metadata = &md;
        req.feature_order = c.feature_order.as_deref();
        req.output_order = c.output_order.as_deref();
        req.compressed = c.compressed;
        $zpb::bake(&req).map_err(|e| format!("{e:?}"))
    }};
}

struct LayerCase {
    in_dim: usize,
    out_dim: usize,
    act: u8,
    dtype: u8,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

struct Case {
    schema: u64,
    mean: Vec<f32>,
    scale: Vec<f32>,
    layers: Vec<LayerCase>,
    bounds: Vec<(f32, f32)>,
    specs: Vec<(f32, f32)>,
    sparse: Vec<(u32, f32)>,
    transforms: Option<String>,
    feature_order: Option<Vec<u32>>,
    output_order: Option<Vec<u32>>,
    compressed: bool,
}

fn gen_case(rng: &mut Rng, k: u64) -> Case {
    let n_layers = 1 + rng.below(4);
    let mut dims = vec![DIMS[rng.below(DIMS.len())]];
    for _ in 0..n_layers {
        dims.push(DIMS[rng.below(DIMS.len())]);
    }
    let n_in = dims[0];
    let n_out = *dims.last().unwrap();
    let mut layers = Vec::new();
    for w in dims.windows(2) {
        let (i, o) = (w[0], w[1]);
        // Some exact zeros so the `s == 0.0` skip and dead units are hit.
        let weights = (0..i * o)
            .map(|_| {
                if rng.below(6) == 0 {
                    0.0
                } else {
                    rng.range(-0.8, 0.8)
                }
            })
            .collect();
        layers.push(LayerCase {
            in_dim: i,
            out_dim: o,
            act: rng.below(3) as u8,
            dtype: rng.below(3) as u8,
            weights,
            biases: (0..o).map(|_| rng.range(-0.5, 0.5)).collect(),
        });
    }
    let mean = (0..n_in).map(|_| rng.range(-2.0, 2.0)).collect();
    let scale = (0..n_in)
        .map(|_| {
            if rng.below(10) == 0 {
                0.0
            } else {
                rng.range(0.2, 3.0)
            }
        })
        .collect();
    let bounds = if rng.below(2) == 0 {
        (0..n_in).map(|_| (-5.0, 5.0)).collect()
    } else {
        vec![]
    };
    let specs = if rng.below(3) == 0 {
        (0..n_out).map(|_| (-1.0, 1.0)).collect()
    } else {
        vec![]
    };
    let sparse = if rng.below(3) == 0 {
        vec![(rng.below(n_out) as u32, rng.range(-1.0, 1.0))]
    } else {
        vec![]
    };
    let transforms = (rng.below(3) == 0).then(|| {
        (0..n_in)
            .map(|_| ["identity", "log1p", "signed_log1p", "signed_cbrt"][rng.below(4)])
            .collect::<Vec<_>>()
            .join("\n")
    });
    let feature_order = (rng.below(3) == 0).then(|| perm(rng, n_in));
    let output_order = (rng.below(3) == 0).then(|| perm(rng, n_out));
    Case {
        schema: k,
        mean,
        scale,
        layers,
        bounds,
        specs,
        sparse,
        transforms,
        feature_order,
        output_order,
        compressed: rng.below(2) == 0,
    }
}

fn synth(cases: u64) -> ExitCode {
    let n = n_vectors().min(400);
    let mut rng = Rng(0x5a6e_7a91_d00d_f00d);
    let (mut bad, mut values, mut vectors, mut baked) = (0u64, 0usize, 0usize, 0u64);
    for k in 0..cases {
        let case = gen_case(&mut rng, k);
        let old = bake_with!(zp_old, zpb_old, &case);
        let new = bake_with!(zp_new, zpb_new, &case);
        if old != new {
            bad += 1;
            println!(
                "case {k}: composer output differs (old {} bytes / new {} bytes, ok={}/{})",
                old.as_ref().map(Vec::len).unwrap_or(0),
                new.as_ref().map(Vec::len).unwrap_or(0),
                old.is_ok(),
                new.is_ok()
            );
            continue;
        }
        let Ok(bytes) = old else {
            continue;
        };
        baked += 1;
        let c = compare(&bytes, n, k);
        values += c.values;
        vectors += c.vectors;
        if !c.mismatches.is_empty() {
            bad += 1;
            println!("case {k}: runtime differs");
            for m in &c.mismatches {
                println!("    {m}");
            }
        }
    }
    println!(
        "# synth cases={cases} baked={baked} vectors={vectors} values={values} mismatched={bad}"
    );
    if bad == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("check") if args.len() > 1 => check(&args[1..]),
        Some("synth") => synth(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500)),
        _ => {
            eprintln!("usage: zenpredict-graph-parity check <bake.bin>... | synth [cases]");
            ExitCode::from(2)
        }
    }
}
