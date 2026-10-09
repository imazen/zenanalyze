//! ZNPR v4 op-graph tests: the gated-head gate property, random graphs
//! against a naive reference evaluator, chain-as-graph equivalence,
//! JSON baking, typed node views, and every load-time rejection rule.

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use zenpredict::{Activation, Model, NodeEntry, NodeView, PredictError, Predictor, WeightDtype};
use zenpredict_bake::{
    BakeError, BakeJsonError, BakeLayer, BakeNode, BakeRequest, GraphBakeRequest, bake,
    bake_from_json_str, bake_graph,
};

const GATED: &str = include_str!("../examples/gated_head.json");

// ───────────────────────── reference evaluator ─────────────────────────

// The per-node reference evaluator (`reference_eval`) is shared with the
// differential fuzz target, so tests and fuzzing check the executor
// against one definition of the spec.
include!("../../zenpredict/fuzz/fuzz_targets/graph_core.rs");

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn gated_bytes() -> Vec<u8> {
    bake_from_json_str(GATED).expect("bake gated_head.json")
}

// ───────────────────────── gated head ─────────────────────────

/// The E33 arm-B property: with the difference slice `d` at zero, the
/// gated head is exactly `+0.0` for any finite reference inputs `r` —
/// including magnitudes that overflow to ±inf after the scaler.
#[test]
fn gated_head_is_exactly_zero_when_d_is_zero() {
    let bytes = gated_bytes();
    let model = Model::from_bytes(&bytes).unwrap();
    let mut p = Predictor::new(&model);
    let mut rng = SmallRng::seed_from_u64(0x9a7e);
    let mut checked = 0;
    for k in 0..20_000 {
        let r: [f32; 3] = core::array::from_fn(|_| match k % 3 {
            // Any finite bit pattern: subnormals to ±f32::MAX.
            0 => loop {
                let f = f32::from_bits(rng.random::<u32>());
                if f.is_finite() {
                    break f;
                }
            },
            1 => rng.random_range(-50.0f32..50.0),
            _ => rng.random_range(-1.0e30f32..1.0e30),
        });
        let d_sign = if k % 2 == 0 { 0.0 } else { -0.0 };
        let x = [d_sign, d_sign, d_sign, r[0], r[1], r[2]];
        let g = p.predict(&x).unwrap()[0];
        assert_eq!(
            g.to_bits(),
            0.0f32.to_bits(),
            "x={x:?} gave {g} ({:#010x}), want +0.0",
            g.to_bits()
        );
        checked += 1;
    }
    assert_eq!(checked, 20_000);
}

/// Away from `d = 0` the head matches the reference evaluator bit for
/// bit, and is non-zero for generic inputs.
#[test]
fn gated_head_matches_reference() {
    let bytes = gated_bytes();
    let model = Model::from_bytes(&bytes).unwrap();
    assert_eq!(model.version(), zenpredict::GRAPH_FORMAT_VERSION);
    assert_eq!(model.n_nodes(), 7);
    assert!(!model.is_layer_chain());
    assert_eq!(model.n_layers(), 3, "three Dense nodes");
    let mut p = Predictor::new(&model);
    let mut rng = SmallRng::seed_from_u64(0x9a7f);
    let mut nonzero = 0;
    for _ in 0..5_000 {
        let x: [f32; 6] = core::array::from_fn(|_| rng.random_range(-4.0f32..4.0));
        let got = p.predict(&x).unwrap()[0];
        let want = reference_eval(&model, &x)[0];
        assert_eq!(got.to_bits(), want.to_bits(), "x={x:?}");
        nonzero += usize::from(got != 0.0);
    }
    assert!(nonzero > 4_000, "only {nonzero}/5000 non-zero outputs");
    // Spot value: r at its mean (scaled 0, exp = 1), d = (0.5, -0.25, 1):
    // relu(w·d) = (0.9, 0, 0, 0.75) → 1.5·0.9 + 0.5·0.75 = 1.725.
    let g = p.predict(&[0.5, -0.25, 1.0, 0.5, -1.0, 2.0]).unwrap()[0];
    assert!((g - 1.725).abs() < 1e-6, "g = {g}");
}

#[test]
fn gated_head_node_views() {
    let bytes = gated_bytes();
    let model = Model::from_bytes(&bytes).unwrap();
    let kinds: Vec<String> = model
        .nodes()
        .map(|n| match n {
            NodeView::Input { width, .. } => format!("in{width}"),
            NodeView::Gather { input, indices, .. } => format!("g{input}{indices:?}"),
            NodeView::Dense { input, layer, .. } => format!(
                "d{input}:{}x{}:{:?}:{}",
                layer.in_dim,
                layer.out_dim,
                layer.activation,
                layer.biases.len()
            ),
            NodeView::Mul { a, b, width, .. } => format!("m{a},{b}:{width}"),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "in6",
            "g0[0, 1, 2]",
            "g0[3, 4, 5]",
            "d1:3x4:Relu:0",
            "d2:3x4:Exp:0",
            "m3,4:4",
            "d5:4x1:Identity:0"
        ]
    );
    let dims: Vec<(usize, usize)> = model.layers().map(|l| (l.in_dim, l.out_dim)).collect();
    assert_eq!(dims, [(3, 4), (3, 4), (4, 1)]);
}

#[test]
fn predictor_try_new_matches_new() {
    let bytes = gated_bytes();
    let model = Model::from_bytes(&bytes).unwrap();
    let mut a = Predictor::new(&model);
    let mut b = Predictor::try_new(&model).unwrap();
    let x = [0.3, -0.7, 1.1, 2.0, -0.5, 0.25];
    assert_eq!(bits(a.predict(&x).unwrap()), bits(b.predict(&x).unwrap()));
}

// ───────────────────────── random graphs ─────────────────────────

struct OwnedNode {
    kind: u8,
    a: u32,
    b: u32,
    out_dim: usize,
    act: Activation,
    dtype: WeightDtype,
    weights: Vec<f32>,
    biases: Option<Vec<f32>>,
    list: Vec<u32>,
}

impl OwnedNode {
    fn new(kind: u8) -> Self {
        Self {
            kind,
            a: 0,
            b: 0,
            out_dim: 0,
            act: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: vec![],
            biases: None,
            list: vec![],
        }
    }
    fn borrow(&self) -> BakeNode<'_> {
        match self.kind {
            0 => BakeNode::input(self.out_dim),
            1 => BakeNode::dense(
                self.a,
                self.out_dim,
                self.act,
                self.dtype,
                &self.weights,
                self.biases.as_deref(),
            ),
            2 => BakeNode::activation(self.a, self.act),
            3 => BakeNode::gather(self.a, &self.list),
            4 => BakeNode::add(self.a, self.b),
            5 => BakeNode::mul(self.a, self.b),
            _ => BakeNode::concat(&self.list),
        }
    }
}

const ACTS: [Activation; 5] = [
    Activation::Identity,
    Activation::Relu,
    Activation::LeakyRelu,
    Activation::Exp,
    Activation::Softplus,
];
const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::F16, WeightDtype::I8];

/// A random DAG over every op, every activation and every dtype, with
/// fan-out, repeated inputs (`Mul(a, a)`), contiguous and scattered
/// gathers, and a final Concat + Dense collecting every otherwise-unread
/// node, so no node is dead.
fn random_graph(rng: &mut SmallRng) -> (Vec<OwnedNode>, Vec<f32>, Vec<f32>) {
    let n_in = rng.random_range(1..24usize);
    let mut nodes = vec![{
        let mut n = OwnedNode::new(0);
        n.out_dim = n_in;
        n
    }];
    let mut widths = vec![n_in];
    let body = rng.random_range(1..14usize);
    for _ in 0..body {
        let i = nodes.len();
        let pick = |rng: &mut SmallRng| rng.random_range(0..i) as u32;
        let kind = rng.random_range(1..7u8);
        let mut n = OwnedNode::new(kind);
        match kind {
            1 => {
                n.a = pick(rng);
                n.out_dim = rng.random_range(1..20usize);
                n.act = ACTS[rng.random_range(0..5)];
                n.dtype = DTYPES[rng.random_range(0..3)];
                let in_dim = widths[n.a as usize];
                n.weights = (0..in_dim * n.out_dim)
                    .map(|_| {
                        if rng.random_range(0..5) == 0 {
                            0.0
                        } else {
                            rng.random_range(-0.9f32..0.9)
                        }
                    })
                    .collect();
                if rng.random_bool(0.6) {
                    n.biases = Some(
                        (0..n.out_dim)
                            .map(|_| rng.random_range(-0.5f32..0.5))
                            .collect(),
                    );
                }
            }
            2 => {
                n.a = pick(rng);
                n.act = ACTS[rng.random_range(0..5)];
                n.out_dim = widths[n.a as usize];
            }
            3 => {
                n.a = pick(rng);
                let src = widths[n.a as usize];
                n.out_dim = rng.random_range(1..20usize);
                if rng.random_bool(0.4) && n.out_dim <= src {
                    let start = rng.random_range(0..=src - n.out_dim) as u32;
                    n.list = (start..start + n.out_dim as u32).collect();
                } else {
                    n.list = (0..n.out_dim)
                        .map(|_| rng.random_range(0..src) as u32)
                        .collect();
                }
            }
            4 | 5 => {
                n.a = pick(rng);
                let w = widths[n.a as usize];
                let same: Vec<u32> = (0..i as u32).filter(|&j| widths[j as usize] == w).collect();
                n.b = same[rng.random_range(0..same.len())];
                n.out_dim = w;
            }
            _ => {
                let k = rng.random_range(1..4usize);
                n.list = (0..k).map(|_| pick(rng)).collect();
                n.out_dim = n.list.iter().map(|&j| widths[j as usize]).sum();
                if n.out_dim > 256 {
                    // Keep nested concats from growing geometrically.
                    n.list.truncate(1);
                    n.out_dim = widths[n.list[0] as usize];
                }
            }
        }
        widths.push(n.out_dim);
        nodes.push(n);
    }
    // Collect every node nobody reads into a Concat, then a final Dense.
    let mut read = vec![false; nodes.len()];
    for n in &nodes {
        match n.kind {
            0 => {}
            1..=3 => read[n.a as usize] = true,
            4 | 5 => {
                read[n.a as usize] = true;
                read[n.b as usize] = true;
            }
            _ => n.list.iter().for_each(|&j| read[j as usize] = true),
        }
    }
    let loose: Vec<u32> = (0..nodes.len() as u32)
        .filter(|&j| !read[j as usize])
        .collect();
    let mut cat = OwnedNode::new(6);
    cat.out_dim = loose.iter().map(|&j| widths[j as usize]).sum();
    cat.list = loose;
    let cat_idx = nodes.len() as u32;
    let cat_w = cat.out_dim;
    nodes.push(cat);
    let mut head = OwnedNode::new(1);
    head.a = cat_idx;
    head.out_dim = rng.random_range(1..6usize);
    head.act = ACTS[rng.random_range(0..3)];
    head.dtype = DTYPES[rng.random_range(0..3)];
    head.weights = (0..cat_w * head.out_dim)
        .map(|_| rng.random_range(-0.5f32..0.5))
        .collect();
    head.biases = Some(vec![0.125; head.out_dim]);
    nodes.push(head);
    let mean = (0..n_in).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let scale = (0..n_in)
        .map(|_| {
            if rng.random_range(0..8) == 0 {
                0.0
            } else {
                rng.random_range(0.3f32..2.0)
            }
        })
        .collect();
    (nodes, mean, scale)
}

/// The executor's liveness-packed arena against a reference that gives
/// every node its own buffer: any slot-planning bug (an output written
/// over a still-live input) shows up as a bit difference.
#[test]
fn random_graphs_match_reference_evaluator() {
    let mut rng = SmallRng::seed_from_u64(0x6a9f_0001);
    let (mut graphs, mut values) = (0, 0usize);
    for g in 0..400 {
        let (nodes, mean, scale) = random_graph(&mut rng);
        let borrowed: Vec<BakeNode<'_>> = nodes.iter().map(OwnedNode::borrow).collect();
        let req = GraphBakeRequest::new(g, &mean, &scale, &borrowed);
        let bytes = bake_graph(&req).unwrap_or_else(|e| panic!("graph {g}: {e}"));
        let model = Model::from_bytes(&bytes).unwrap();
        let mut p = Predictor::new(&model);
        for _ in 0..40 {
            let x: Vec<f32> = (0..model.n_inputs())
                .map(|_| {
                    if rng.random_range(0..10) == 0 {
                        0.0
                    } else {
                        rng.random_range(-3.0f32..3.0)
                    }
                })
                .collect();
            let got = bits(p.predict(&x).unwrap());
            let want = bits(&reference_eval(&model, &x));
            assert_eq!(got, want, "graph {g}, x={x:?}");
            values += got.len();
        }
        graphs += 1;
    }
    assert_eq!(graphs, 400);
    assert!(values > 16_000);
}

/// Compressed graph bakes load and predict identically.
#[test]
fn compressed_graph_matches_uncompressed() {
    let mut rng = SmallRng::seed_from_u64(0x6a9f_0002);
    for g in 0..40 {
        let (nodes, mean, scale) = random_graph(&mut rng);
        let borrowed: Vec<BakeNode<'_>> = nodes.iter().map(OwnedNode::borrow).collect();
        let req = GraphBakeRequest::new(g, &mean, &scale, &borrowed);
        let plain = bake_graph(&req).unwrap();
        let packed = req.compressed(true).bake().unwrap();
        let (a, b) = (
            Model::from_bytes(&plain).unwrap(),
            Model::from_bytes(&packed).unwrap(),
        );
        let (mut pa, mut pb) = (Predictor::new(&a), Predictor::new(&b));
        let x: Vec<f32> = (0..a.n_inputs()).map(|i| i as f32 * 0.37 - 1.0).collect();
        assert_eq!(bits(pa.predict(&x).unwrap()), bits(pb.predict(&x).unwrap()));
    }
}

// ───────────────────────── chain as graph ─────────────────────────

/// A v3 chain and the same network written as a v4 `Input → Dense…`
/// graph produce bit-identical outputs (identity hidden-unit order so the
/// two files hold the same weights in the same order).
#[test]
fn chain_and_equivalent_graph_agree() {
    let mut rng = SmallRng::seed_from_u64(0x6a9f_0003);
    for case in 0..200 {
        let n_layers = rng.random_range(1..5usize);
        let dims: Vec<usize> = (0..=n_layers)
            .map(|_| rng.random_range(1..40usize))
            .collect();
        let ws: Vec<Vec<f32>> = dims
            .windows(2)
            .map(|d| {
                (0..d[0] * d[1])
                    .map(|_| rng.random_range(-0.7f32..0.7))
                    .collect()
            })
            .collect();
        let bs: Vec<Vec<f32>> = dims[1..]
            .iter()
            .map(|&o| (0..o).map(|_| rng.random_range(-0.3f32..0.3)).collect())
            .collect();
        let acts: Vec<Activation> = (0..n_layers)
            .map(|_| ACTS[rng.random_range(0..3)])
            .collect();
        let dts: Vec<WeightDtype> = (0..n_layers)
            .map(|_| DTYPES[rng.random_range(0..3)])
            .collect();
        let layers: Vec<BakeLayer<'_>> = (0..n_layers)
            .map(|k| BakeLayer {
                in_dim: dims[k],
                out_dim: dims[k + 1],
                activation: acts[k],
                dtype: dts[k],
                weights: &ws[k],
                biases: &bs[k],
            })
            .collect();
        let mean: Vec<f32> = (0..dims[0])
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect();
        let scale: Vec<f32> = (0..dims[0])
            .map(|_| rng.random_range(0.5f32..2.0))
            .collect();
        let ident: Vec<Vec<u32>> = dims[1..n_layers]
            .iter()
            .map(|&d| (0..d as u32).collect())
            .collect();
        let ident_refs: Vec<&[u32]> = ident.iter().map(Vec::as_slice).collect();
        let mut chain_req = BakeRequest::new(case, 0, &mean, &scale, &layers);
        chain_req.hu_permutations = Some(&ident_refs);
        let chain = bake(&chain_req).unwrap();

        let mut nodes = vec![BakeNode::input(dims[0])];
        for k in 0..n_layers {
            nodes.push(BakeNode::dense(
                k as u32,
                dims[k + 1],
                acts[k],
                dts[k],
                &ws[k],
                Some(&bs[k]),
            ));
        }
        let graph = bake_graph(&GraphBakeRequest::new(case, &mean, &scale, &nodes)).unwrap();

        let (mc, mg) = (
            Model::from_bytes(&chain).unwrap(),
            Model::from_bytes(&graph).unwrap(),
        );
        assert_eq!(mc.version(), 3);
        assert_eq!(mg.version(), 4);
        assert!(mc.is_layer_chain() && mg.is_layer_chain());
        assert_eq!(mc.n_nodes(), n_layers + 1);
        assert_eq!(mc.n_layers(), mg.n_layers());
        let (mut pc, mut pg) = (Predictor::new(&mc), Predictor::new(&mg));
        for _ in 0..30 {
            let x: Vec<f32> = (0..dims[0])
                .map(|_| rng.random_range(-3.0f32..3.0))
                .collect();
            assert_eq!(
                bits(pc.predict(&x).unwrap()),
                bits(pg.predict(&x).unwrap()),
                "case {case}"
            );
        }
    }
}

// ───────────────────────── baking errors ─────────────────────────

#[test]
fn chain_bake_refuses_v4_only_activations() {
    for act in [Activation::Exp, Activation::Softplus] {
        let w = [1.0f32, 2.0];
        let b = [0.0f32];
        let layers = [BakeLayer {
            in_dim: 2,
            out_dim: 1,
            activation: act,
            dtype: WeightDtype::F32,
            weights: &w,
            biases: &b,
        }];
        let err = bake(&BakeRequest::new(0, 0, &[0.0; 2], &[1.0; 2], &layers)).unwrap_err();
        assert_eq!(err, BakeError::ChainActivationUnsupported { layer: 0 });
    }
}

#[test]
fn json_graph_rules() {
    let with = |field: &str| GATED.replacen('{', &format!("{{ {field},"), 1);
    // graph + optimize is refused.
    let err = bake_from_json_str(&with(r#""optimize": true"#)).unwrap_err();
    assert!(
        matches!(
            err,
            BakeJsonError::Bake(BakeError::GraphInvalid { node: 0, .. })
        ),
        "{err}"
    );
    // graph + non-empty layers is refused.
    let layers = r#""layers": [{"in_dim":6,"out_dim":1,"activation":"identity","dtype":"f32","weights":[0,0,0,0,0,0],"biases":[0]}]"#;
    let err = bake_from_json_str(&with(layers)).unwrap_err();
    assert!(
        matches!(
            err,
            BakeJsonError::Bake(BakeError::GraphInvalid { node: 0, .. })
        ),
        "{err}"
    );
    // compressed + zerobias apply to graphs and keep the gate property.
    let bytes = bake_from_json_str(&with(r#""compressed": true, "zerobias_tau": 0.005"#)).unwrap();
    let model = Model::from_bytes(&bytes).unwrap();
    let mut p = Predictor::new(&model);
    assert_eq!(
        p.predict(&[0.0, 0.0, 0.0, 7.0, -3.0, 1.0e20]).unwrap()[0].to_bits(),
        0.0f32.to_bits()
    );
}

#[test]
fn bake_graph_reports_composition_errors() {
    let mean = [0.0f32; 2];
    let scale = [1.0f32; 2];
    let req = |nodes: &[BakeNode<'_>]| -> Result<Vec<u8>, BakeError> {
        GraphBakeRequest::new(0, &mean, &scale, nodes).bake()
    };
    let w = [0.5f32; 2];
    // Forward reference.
    let nodes = [
        BakeNode::input(2),
        BakeNode::activation(1, Activation::Relu),
    ];
    assert!(matches!(
        req(&nodes),
        Err(BakeError::GraphInvalid { node: 1, .. })
    ));
    // Node 0 not Input.
    let nodes = [BakeNode::add(0, 0)];
    assert!(matches!(
        req(&nodes),
        Err(BakeError::GraphInvalid { node: 0, .. })
    ));
    // Wrong Dense weight length.
    let nodes = [
        BakeNode::input(2),
        BakeNode::dense(0, 3, Activation::Identity, WeightDtype::F32, &w, None),
    ];
    assert!(matches!(
        req(&nodes),
        Err(BakeError::GraphInvalid { node: 1, .. })
    ));
    // Dead node: the parser's verdict comes back as GraphRejected.
    let nodes = [
        BakeNode::input(2),
        BakeNode::gather(0, &[1]),
        BakeNode::activation(0, Activation::Softplus),
    ];
    assert!(matches!(
        req(&nodes),
        Err(BakeError::GraphRejected(PredictError::GraphMalformed {
            node: 1,
            ..
        }))
    ));
    // Width mismatch on Add.
    let nodes = [
        BakeNode::input(2),
        BakeNode::gather(0, &[1]),
        BakeNode::add(0, 1),
    ];
    assert!(matches!(
        req(&nodes),
        Err(BakeError::GraphRejected(PredictError::GraphShapeMismatch {
            node: 2,
            ..
        }))
    ));
}

// ───────────────────────── load-time rejections ─────────────────────────

const HDR: usize = zenpredict::wire::HEADER_SIZE;
const NE: usize = zenpredict::wire::NODE_ENTRY_SIZE;

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
/// Byte offset of the first input index of node `i`.
fn first_input_at(b: &[u8], i: usize) -> usize {
    u32_at(b, HDR + i * NE + core::mem::offset_of!(NodeEntry, inputs)) as usize
}

fn load_err(mutate: impl FnOnce(&mut Vec<u8>)) -> PredictError {
    let mut b = gated_bytes();
    mutate(&mut b);
    Model::from_bytes(&b).unwrap_err()
}

#[test]
fn unmodified_gated_bytes_load() {
    Model::from_bytes(&gated_bytes()).unwrap();
}

#[test]
fn rejects_unknown_op() {
    assert_eq!(
        load_err(|b| b[HDR + 3 * NE] = 7),
        PredictError::UnknownGraphOp { node: 3, byte: 7 }
    );
    assert_eq!(
        load_err(|b| b[HDR + 5 * NE] = 0xff),
        PredictError::UnknownGraphOp {
            node: 5,
            byte: 0xff
        }
    );
}

#[test]
fn rejects_forward_and_self_references() {
    assert_eq!(
        load_err(|b| {
            let at = first_input_at(b, 3);
            put_u32(b, at, 5);
        }),
        PredictError::GraphInputRef { node: 3, input: 5 }
    );
    assert_eq!(
        load_err(|b| {
            let at = first_input_at(b, 3);
            put_u32(b, at, 3);
        }),
        PredictError::GraphInputRef { node: 3, input: 3 }
    );
}

#[test]
fn rejects_unknown_activation_and_dtype() {
    assert_eq!(
        load_err(|b| b[HDR + 3 * NE + 1] = 5),
        PredictError::UnknownActivation { byte: 5 }
    );
    assert_eq!(
        load_err(|b| b[HDR + 3 * NE + 2] = 3),
        PredictError::UnknownWeightDtype { byte: 3 }
    );
}

#[test]
fn rejects_nonzero_reserved_and_unused_fields() {
    let malformed = |e: PredictError, node: usize| {
        assert!(
            matches!(e, PredictError::GraphMalformed { node: n, .. } if n == node),
            "{e:?}"
        );
    };
    malformed(load_err(|b| b[HDR + 3 * NE + 3] = 1), 3); // flags
    malformed(load_err(|b| b[HDR + 4 * NE + 44] = 1), 4); // reserved
    malformed(load_err(|b| b[HDR + 5 * NE + 1] = 1), 5); // activation on Mul
    malformed(load_err(|b| b[HDR + NE + 2] = 1), 1); // dtype on Gather
    malformed(load_err(|b| b[HDR + NE + 1] = 1), 1); // activation on Gather
}

#[test]
fn rejects_wrong_arity_and_misplaced_input() {
    let malformed = |e: PredictError, node: usize| {
        assert!(
            matches!(e, PredictError::GraphMalformed { node: n, .. } if n == node),
            "{e:?}"
        );
    };
    // Mul with one input (inputs len 8 → 4).
    malformed(
        load_err(|b| {
            put_u32(
                b,
                HDR + 5 * NE + core::mem::offset_of!(NodeEntry, inputs) + 4,
                4,
            )
        }),
        5,
    );
    // Node 0 is not Input.
    malformed(load_err(|b| b[HDR] = zenpredict::wire::OP_ACTIVATION), 0);
    // A second Input node.
    malformed(load_err(|b| b[HDR + NE] = zenpredict::wire::OP_INPUT), 1);
}

#[test]
fn rejects_shape_mismatches() {
    // Mul declared width 3, inputs are 4 wide.
    assert_eq!(
        load_err(|b| put_u32(b, HDR + 5 * NE + 4, 3)),
        PredictError::GraphShapeMismatch {
            node: 5,
            expected: 3,
            got: 4
        }
    );
    // Header n_outputs disagrees with the last node.
    assert_eq!(
        load_err(|b| put_u32(b, 12, 2)),
        PredictError::OutputDimMismatch {
            expected: 2,
            got: 1
        }
    );
    // Input width disagrees with header n_inputs: also mismatched scaler,
    // but the node table is checked first.
    assert_eq!(
        load_err(|b| put_u32(b, HDR + 4, 5)),
        PredictError::GraphShapeMismatch {
            node: 0,
            expected: 6,
            got: 5
        }
    );
}

#[test]
fn rejects_gather_index_out_of_range() {
    let e = load_err(|b| {
        let at = u32_at(b, HDR + NE + core::mem::offset_of!(NodeEntry, data0)) as usize;
        put_u32(b, at + 4, 6);
    });
    assert!(
        matches!(e, PredictError::GraphMalformed { node: 1, .. }),
        "{e:?}"
    );
}

#[test]
fn rejects_bad_node_counts_and_v4_permutations() {
    assert!(matches!(
        load_err(|b| put_u32(b, 16, 1)),
        PredictError::GraphMalformed { node: 0, .. }
    ));
    assert_eq!(
        load_err(|b| put_u32(b, 16, 0)),
        PredictError::ZeroDimension { what: "n_nodes" }
    );
    assert!(matches!(
        load_err(|b| put_u32(b, 16, zenpredict::limits::MAX_NODES as u32 + 1)),
        PredictError::DimensionOverflow { .. }
    ));
    // A node-count change without a matching table is a section error.
    assert!(matches!(
        load_err(|b| put_u32(b, 16, 8)),
        PredictError::SectionOutOfRange { .. }
    ));
    // feature_order / output_order must stay empty in v4.
    for off in [
        zenpredict::wire::SECTION_OFF_FEATURE_ORDER,
        zenpredict::wire::SECTION_OFF_OUTPUT_ORDER,
    ] {
        assert!(matches!(
            load_err(|b| {
                put_u32(b, off, 128);
                put_u32(b, off + 4, 6);
            }),
            PredictError::GraphMalformed { node: 0, .. }
        ));
    }
}

#[test]
fn v3_chain_still_rejects_v4_only_activation_bytes() {
    let w = [1.0f32, 2.0];
    let b = [0.0f32];
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &w,
        biases: &b,
    }];
    let mut bytes = bake(&BakeRequest::new(0, 0, &[0.0; 2], &[1.0; 2], &layers)).unwrap();
    for byte in [3u8, 4] {
        bytes[HDR + 8] = byte; // LayerEntry.activation
        assert_eq!(
            Model::from_bytes(&bytes).unwrap_err(),
            PredictError::UnknownActivation { byte }
        );
    }
}

#[test]
fn future_versions_are_rejected() {
    let mut b = gated_bytes();
    b[4..6].copy_from_slice(&5u16.to_le_bytes());
    assert_eq!(
        Model::from_bytes(&b).unwrap_err(),
        PredictError::UnsupportedVersion {
            version: 5,
            expected: 3
        }
    );
}

/// `MAX_TOTAL_WEIGHTS` bounds compute even though the file is small
/// enough (I8 weights, 1 byte each).
#[test]
fn rejects_total_weights_over_limit() {
    let limit = zenpredict::limits::MAX_TOTAL_WEIGHTS;
    let n_in = 4097usize;
    let out = 4096usize;
    assert!(n_in * out > limit && n_in * out < zenpredict::limits::MAX_BAKE_BYTES);
    let w = vec![0.0f32; n_in * out];
    let mean = vec![0.0f32; n_in];
    let scale = vec![1.0f32; n_in];
    let nodes = [
        BakeNode::input(n_in),
        BakeNode::dense(0, out, Activation::Identity, WeightDtype::I8, &w, None),
    ];
    let err = bake_graph(&GraphBakeRequest::new(0, &mean, &scale, &nodes)).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::GraphRejected(PredictError::DimensionOverflow { .. })
        ),
        "{err}"
    );
}

/// `MAX_SCRATCH_ELEMS` bounds the planned arena: 64 full-width gathers
/// live at once alongside the input exceed it.
#[test]
fn rejects_scratch_over_limit() {
    let w = zenpredict::limits::MAX_DIM;
    let k = zenpredict::limits::MAX_SCRATCH_ELEMS / w; // 64
    let idx: Vec<u32> = (0..w as u32).collect();
    let mean = vec![0.0f32; w];
    let scale = vec![1.0f32; w];
    let mut nodes = vec![BakeNode::input(w)];
    for _ in 0..k {
        nodes.push(BakeNode::gather(0, &idx));
    }
    // Pairwise Add tree down to one node.
    let mut level: Vec<u32> = (1..=k as u32).collect();
    while level.len() > 1 {
        let mut next = Vec::new();
        for pair in level.chunks(2) {
            nodes.push(BakeNode::add(pair[0], pair[1]));
            next.push(nodes.len() as u32 - 1);
        }
        level = next;
    }
    let err = bake_graph(&GraphBakeRequest::new(0, &mean, &scale, &nodes)).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::GraphRejected(PredictError::DimensionOverflow { .. })
        ),
        "{err}"
    );
}

// ───────────────────────── compute budget ─────────────────────────

/// Review case: a ~577 KB bake of Input(65,536) followed by 1,023 Exp
/// activations passed every v4 check and took ~210 ms per predict.
/// `MAX_TOTAL_ELEMS` (op-weighted width, Exp = 32 per element) refuses it.
#[test]
fn rejects_elementwise_work_over_limit() {
    let w = zenpredict::limits::MAX_DIM;
    let mean = vec![0.0f32; w];
    let scale = vec![1.0f32; w];
    let mut nodes = vec![BakeNode::input(w)];
    for i in 0..zenpredict::limits::MAX_NODES - 1 {
        nodes.push(BakeNode::activation(i as u32, Activation::Exp));
    }
    let err = bake_graph(&GraphBakeRequest::new(0, &mean, &scale, &nodes)).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::GraphRejected(PredictError::DimensionOverflow { what })
                if what.contains("MAX_TOTAL_ELEMS")
        ),
        "{err}"
    );
}

/// The same shape under the budget loads and runs:
/// 1,024 + 100 × 1,024 × 32 = 3,277,824 ≤ 2^22.
#[test]
fn elementwise_work_under_limit_loads() {
    let w = 1024usize;
    let mean = vec![0.0f32; w];
    let scale = vec![1.0f32; w];
    let mut nodes = vec![BakeNode::input(w)];
    for i in 0..100u32 {
        nodes.push(BakeNode::activation(i, Activation::Exp));
    }
    let bytes = bake_graph(&GraphBakeRequest::new(0, &mean, &scale, &nodes)).unwrap();
    let model = Model::from_bytes(&bytes).unwrap();
    let x = vec![0.25f32; w];
    let got = Predictor::new(&model).predict(&x).unwrap().to_vec();
    assert_eq!(bits(&got), bits(&reference_eval(&model, &x)));
}

// ───────────────────── more load-time rejections ─────────────────────

const OFF_INPUTS: usize = core::mem::offset_of!(NodeEntry, inputs);
const OFF_DATA0: usize = core::mem::offset_of!(NodeEntry, data0);
const OFF_DATA1: usize = core::mem::offset_of!(NodeEntry, data1);
const OFF_DATA2: usize = core::mem::offset_of!(NodeEntry, data2);

/// Input(4) → Gather[0,1] → Gather[2,3] → Concat(1,2) → Dense(I8, 4→2).
fn concat_i8_bytes() -> Vec<u8> {
    let w = [0.5f32, -0.25, 0.75, 0.1, -0.6, 0.3, 0.2, -0.9];
    let b = [0.0f32, 0.125];
    let nodes = [
        BakeNode::input(4),
        BakeNode::gather(0, &[0, 1]),
        BakeNode::gather(0, &[2, 3]),
        BakeNode::concat(&[1, 2]),
        BakeNode::dense(3, 2, Activation::Identity, WeightDtype::I8, &w, Some(&b)),
    ];
    bake_graph(&GraphBakeRequest::new(0, &[0.0; 4], &[1.0; 4], &nodes)).unwrap()
}

fn load_err_on(mut b: Vec<u8>, mutate: impl FnOnce(&mut Vec<u8>)) -> PredictError {
    mutate(&mut b);
    Model::from_bytes(&b).unwrap_err()
}

fn assert_malformed(e: PredictError, node: usize) {
    assert!(
        matches!(e, PredictError::GraphMalformed { node: n, .. } if n == node),
        "{e:?}"
    );
}

#[test]
fn concat_i8_fixture_loads() {
    Model::from_bytes(&concat_i8_bytes()).unwrap();
}

#[test]
fn rejects_inputs_length_not_multiple_of_four() {
    let e = load_err_on(gated_bytes(), |b| {
        put_u32(b, HDR + 3 * NE + OFF_INPUTS + 4, 6)
    });
    assert_malformed(e, 3);
}

#[test]
fn rejects_scales_on_non_i8_dense() {
    let e = load_err_on(gated_bytes(), |b| {
        let w = u32_at(b, HDR + 3 * NE + OFF_DATA0);
        put_u32(b, HDR + 3 * NE + OFF_DATA1, w);
        put_u32(b, HDR + 3 * NE + OFF_DATA1 + 4, 16);
    });
    assert_malformed(e, 3);
}

#[test]
fn rejects_wrong_bias_and_scale_lengths() {
    // Bias section of 2 floats on a 4-wide Dense (gated node 3).
    let e = load_err_on(gated_bytes(), |b| {
        let w = u32_at(b, HDR + 3 * NE + OFF_DATA0);
        put_u32(b, HDR + 3 * NE + OFF_DATA2, w);
        put_u32(b, HDR + 3 * NE + OFF_DATA2 + 4, 8);
    });
    assert!(matches!(e, PredictError::SectionOutOfRange { .. }), "{e:?}");
    // I8 scales one float short (concat fixture node 4, out_dim 2).
    let e = load_err_on(concat_i8_bytes(), |b| {
        put_u32(b, HDR + 4 * NE + OFF_DATA1 + 4, 4);
    });
    assert!(matches!(e, PredictError::SectionOutOfRange { .. }), "{e:?}");
}

#[test]
fn rejects_misaligned_or_short_node_table() {
    let e = load_err_on(gated_bytes(), |b| put_u32(b, 48, HDR as u32 + 1));
    assert!(
        matches!(
            e,
            PredictError::SectionMisaligned {
                what: "node_table",
                ..
            }
        ),
        "{e:?}"
    );
    let e = load_err_on(gated_bytes(), |b| put_u32(b, 52, (6 * NE) as u32));
    assert!(matches!(e, PredictError::SectionOutOfRange { .. }), "{e:?}");
}

#[test]
fn rejects_concat_arity_zero_or_over_limit() {
    let e = load_err_on(concat_i8_bytes(), |b| {
        put_u32(b, HDR + 3 * NE + OFF_INPUTS + 4, 0)
    });
    assert_malformed(e, 3);
    let over = (zenpredict::limits::MAX_NODE_INPUTS + 1) * 4;
    let e = load_err_on(concat_i8_bytes(), |b| {
        put_u32(b, HDR + 3 * NE + OFF_INPUTS + 4, over as u32)
    });
    assert_malformed(e, 3);
}

#[test]
fn rejects_concat_width_sum_mismatch() {
    assert_eq!(
        load_err_on(concat_i8_bytes(), |b| put_u32(b, HDR + 3 * NE + 4, 5)),
        PredictError::GraphShapeMismatch {
            node: 3,
            expected: 4,
            got: 5
        }
    );
}

#[test]
fn json_graph_nodes_reject_unknown_keys() {
    // `"bias"` (typo for `"biases"`) must not silently bake a bias-free Dense.
    let bad = GATED.replacen(
        r#""weights": [1.5, -0.75, 2.0, 0.5] }"#,
        r#""weights": [1.5, -0.75, 2.0, 0.5], "bias": [0.0] }"#,
        1,
    );
    assert_ne!(bad, GATED, "fixture text changed; update the splice");
    let err = bake_from_json_str(&bad).unwrap_err();
    assert!(err.to_string().contains("bias"), "{err}");
}
