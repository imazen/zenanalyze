//! Deterministic workload shared by lifecycle benchmarks and heaptrack driver.
use zenpredict::input_program::{DamageShape, InputOp, InputProgram, ReferenceGate};
use zenpredict::{Activation, WeightDtype};
use zenpredict_bake::input_program::bake_with_input_program;
use zenpredict_bake::{BakeLayer, BakeRequest, bake};
pub const CASES: &[&str] = &[
    "legacy",
    "joint_relu",
    "joint_softplus",
    "separate_softplus",
    "drop_softplus",
];
pub struct Fixture {
    pub bytes: Vec<u8>,
    pub source: Vec<f32>,
}
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216. - 0.5) * 0.05
        })
        .collect()
}
pub fn fixture(name: &str) -> Fixture {
    assert!(CASES.contains(&name));
    let n = 984;
    let h = 128;
    let mid = 64;
    let activation = if name.contains("softplus") {
        Activation::Softplus
    } else {
        Activation::Relu
    };
    let w0 = values(n * h, 1);
    let w1 = values(h * mid, 2);
    let w2 = values(mid, 3);
    let layers = [
        BakeLayer {
            in_dim: n,
            out_dim: h,
            activation,
            dtype: WeightDtype::F32,
            weights: &w0,
            biases: &vec![0.1; h],
        },
        BakeLayer {
            in_dim: h,
            out_dim: mid,
            activation,
            dtype: WeightDtype::F32,
            weights: &w1,
            biases: &vec![0.1; mid],
        },
        BakeLayer {
            in_dim: mid,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w2,
            biases: &[1.],
        },
    ];
    let means = vec![0.; n];
    let scales = vec![1.; n];
    let mut req = BakeRequest::new(0x98412864, 0, &means, &scales, &layers);
    req.compressed = true;
    if name == "legacy" {
        return Fixture {
            bytes: bake(&req).unwrap(),
            source: vec![0.25; n],
        };
    }
    let mut nodes = Vec::new();
    let mut outputs = Vec::new();
    for i in 0..n as u32 {
        let d = nodes.len() as u32;
        nodes.push(InputOp::Source { index: i });
        let reference = nodes.len() as u32;
        nodes.push(InputOp::Source {
            index: n as u32 + i,
        });
        let shape = DamageShape::new([0.1, 0.8], [0.5, 1.2, 0.8]).unwrap();
        let output = if name == "separate_softplus" {
            let s = nodes.len() as u32;
            nodes.push(InputOp::Shape { input: d, shape });
            let g = nodes.len() as u32;
            nodes.push(InputOp::Gate {
                input: reference,
                gate: ReferenceGate::new([0.1, 0.7, 1.]).unwrap(),
            });
            let o = nodes.len() as u32;
            nodes.push(InputOp::Product { left: s, right: g });
            o
        } else {
            let p = nodes.len() as u32;
            nodes.push(InputOp::Product {
                left: d,
                right: reference,
            });
            let o = nodes.len() as u32;
            nodes.push(InputOp::Shape { input: p, shape });
            o
        };
        outputs.push(output);
    }
    let mut metadata = Vec::new();
    let transforms;
    if name == "drop_softplus" {
        outputs.insert(0, 0);
        transforms = "drop\n".to_owned() + &vec!["identity"; n].join("\n");
        metadata.push(zenpredict_bake::BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORMS,
            kind: zenpredict::MetadataType::Utf8,
            value: transforms.as_bytes(),
        });
        req.metadata = &metadata;
    }
    let program = InputProgram::new((n * 2) as u32, nodes, outputs).unwrap();
    Fixture {
        bytes: bake_with_input_program(&req, &program).unwrap(),
        source: vec![0.25; n * 2],
    }
}
