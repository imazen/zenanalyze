//! Shared synthetic ZNPR v3 bakes for the zenpredict-viz integration
//! tests: composed in-test via `zenpredict_bake`, so no test depends on
//! a sibling checkout. Two layers, odd widths (SAXPY tail lanes), one
//! zero-variance scaler column, optional `feature_transforms` /
//! `output_specs`, every dtype and activation.
#![allow(dead_code)]

use zenpredict::output_spec::{OutputSpec, OutputTransform};
use zenpredict::{Activation, FeatureBound, MetadataType, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

pub const N_IN: usize = 11; // odd width: exercises the SAXPY 8-lane tail
pub const HIDDEN: usize = 13;
pub const N_OUT: usize = 3;

/// Deterministic xorshift so the bakes are reproducible without a
/// rand dependency.
pub struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        lo + unit * (hi - lo)
    }
}

pub struct Synth {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    pub w0: Vec<f32>,
    pub b0: Vec<f32>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub transforms: String,
    pub specs: Vec<OutputSpec>,
}

impl Synth {
    pub fn new(seed: u64) -> Self {
        let mut r = Lcg(seed | 1);
        let mean: Vec<f32> = (0..N_IN).map(|_| r.next_f32(-2.0, 2.0)).collect();
        // One zero-variance column: exercises the `scale == 0 ⇒ 1.0` rule.
        let mut scale: Vec<f32> = (0..N_IN).map(|_| r.next_f32(0.3, 3.0)).collect();
        scale[4] = 0.0;
        let w0: Vec<f32> = (0..N_IN * HIDDEN).map(|_| r.next_f32(-1.0, 1.0)).collect();
        let b0: Vec<f32> = (0..HIDDEN).map(|_| r.next_f32(-0.5, 0.5)).collect();
        let w1: Vec<f32> = (0..HIDDEN * N_OUT).map(|_| r.next_f32(-1.0, 1.0)).collect();
        let b1: Vec<f32> = (0..N_OUT).map(|_| r.next_f32(-0.5, 0.5)).collect();
        // log1p on every other feature, identity elsewhere (N_IN tokens).
        let transforms = (0..N_IN)
            .map(|i| if i % 2 == 0 { "log1p" } else { "identity" })
            .collect::<Vec<_>>()
            .join("\n");
        let mut specs = vec![OutputSpec::passthrough(); N_OUT];
        specs[0].bounds = FeatureBound {
            low: -0.25,
            high: 0.25,
        };
        specs[1].transform = OutputTransform::Sigmoid as u8;
        Self {
            mean,
            scale,
            w0,
            b0,
            w1,
            b1,
            transforms,
            specs,
        }
    }
}

pub fn synth_bake(
    synth: &Synth,
    dtype: WeightDtype,
    activation: Activation,
    with_transforms: bool,
    with_specs: bool,
) -> Vec<u8> {
    let layers = [
        BakeLayer {
            in_dim: N_IN,
            out_dim: HIDDEN,
            activation,
            dtype,
            weights: &synth.w0,
            biases: &synth.b0,
        },
        BakeLayer {
            in_dim: HIDDEN,
            out_dim: N_OUT,
            activation: Activation::Identity,
            dtype,
            weights: &synth.w1,
            biases: &synth.b1,
        },
    ];
    let names = (0..N_IN)
        .map(|i| format!("feat_synth_{i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut metadata = vec![BakeMetadataEntry {
        key: zenpredict::keys::FEATURE_COLUMNS,
        kind: MetadataType::Utf8,
        value: names.as_bytes(),
    }];
    if with_transforms {
        metadata.push(BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: synth.transforms.as_bytes(),
        });
    }
    let mut b =
        BakeRequest::builder(0x5eed_u64, 0, &synth.mean, &synth.scale, &layers).metadata(&metadata);
    if with_specs {
        b = b.output_specs(&synth.specs);
    }
    bake(&b.build()).expect("bake")
}

pub fn probe_features(seed: u64) -> Vec<f32> {
    // Positive (log1p-safe), one exact zero (the SAXPY zero-skip), one
    // value landing on the zero-variance column.
    let mut r = Lcg(seed | 1);
    let mut f: Vec<f32> = (0..N_IN).map(|_| r.next_f32(0.0, 9.0)).collect();
    f[2] = 0.0;
    f
}

pub fn all_dtypes() -> [WeightDtype; 3] {
    [WeightDtype::F32, WeightDtype::F16, WeightDtype::I8]
}

pub fn all_activations() -> [Activation; 3] {
    [
        Activation::Relu,
        Activation::LeakyRelu,
        Activation::Identity,
    ]
}
