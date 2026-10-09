//! ZNPR v4 static op graph: node-table parsing and validation, v3
//! layer-chain lowering, and liveness-packed scratch planning.
//!
//! Every model — a v4 graph or a v3 chain — runs as a [`Graph`]: a list
//! of nodes in strict topological order (each node reads only earlier
//! nodes), executed in file order by [`crate::inference::forward`]. A v3
//! chain lowers to `Input → Dense → … → Dense`, each Dense carrying the
//! layer's activation fused, so v3 files run the same per-layer code as
//! before.
//!
//! All validation happens here, once, at load: op/activation/dtype bytes,
//! arity, input ordering, per-op widths, Dense section sizes, Gather
//! index ranges, dead nodes, and the [`crate::limits`] bounds on node
//! count, total weights and scratch size. The executor trusts the result.
//!
//! Wire layout: [`crate::wire::NODE_ENTRY_SIZE`]; full spec in
//! `docs/ZNPR_V4_GRAPH.md`.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};

use crate::error::PredictError;
use crate::limits::{MAX_DIM, MAX_NODE_INPUTS, MAX_NODES, MAX_SCRATCH_ELEMS, MAX_TOTAL_WEIGHTS};
use crate::model::{
    Activation, LayerOffsets, LayerView, Section, WeightDtype, cast_f32_section, cast_i8_section,
    cast_u16_section, cast_u32_section, try_vec_with_capacity,
};
use crate::wire::{
    NODE_ENTRY_SIZE, OP_ACTIVATION, OP_ADD, OP_CONCAT, OP_DENSE, OP_GATHER, OP_INPUT, OP_MUL,
};

/// On-disk v4 node-table entry. See [`crate::wire::NODE_ENTRY_SIZE`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct NodeEntry {
    pub(crate) op: u8,
    pub(crate) activation: u8,
    pub(crate) weight_dtype: u8,
    pub(crate) flags: u8,
    pub(crate) out_dim: u32,
    pub(crate) inputs: Section,
    pub(crate) data0: Section,
    pub(crate) data1: Section,
    pub(crate) data2: Section,
    pub(crate) reserved: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<NodeEntry>() == NODE_ENTRY_SIZE);

/// One validated node. Section-backed payloads (Dense weights, Gather
/// indices, Concat inputs) are offsets into the model's owned bytes,
/// re-sliced on demand like v3 layers.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NodeKind {
    Input,
    Dense {
        input: u32,
        layer: LayerOffsets,
    },
    Activation {
        input: u32,
        activation: Activation,
    },
    Gather {
        input: u32,
        indices: Section,
        /// `Some(start)` when the indices are `start, start+1, …`: the
        /// executor copies a contiguous range instead of gathering.
        contiguous_start: Option<u32>,
    },
    Add {
        a: u32,
        b: u32,
    },
    Mul {
        a: u32,
        b: u32,
    },
    Concat {
        inputs: Section,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphNode {
    pub(crate) kind: NodeKind,
    /// Output width of this node.
    pub(crate) width: u32,
    /// Offset of this node's output in the scratch arena. Unused for the
    /// last node, which writes straight into the caller's output buffer.
    pub(crate) slot: u32,
}

/// A validated, planned graph.
#[derive(Debug)]
pub(crate) struct Graph {
    pub(crate) nodes: Vec<GraphNode>,
    /// f32 elements of scratch the executor needs (liveness-packed).
    pub(crate) arena_len: usize,
    /// Node indices of every Dense node, in node order — what
    /// `Model::layers()` walks.
    pub(crate) dense_nodes: Vec<u32>,
    /// True when the graph is `Input → Dense → … → Dense`, each Dense
    /// reading the node before it.
    pub(crate) is_chain: bool,
}

/// Typed view of one graph node, from [`crate::Model::node`].
///
/// Node indices (`input`, `a`, `b`, entries of `inputs`) always refer to
/// earlier nodes. The last node of a model is its output.
#[derive(Debug)]
#[non_exhaustive]
pub enum NodeView<'a> {
    /// The scaled feature vector, `(x - mean) / scale`. Always node 0.
    Input { width: usize },
    /// `act(b + x·W)` over node `input`. `layer.in_dim` is `input`'s
    /// width; `layer.biases` is empty when the node has no bias.
    Dense { input: usize, layer: LayerView<'a> },
    /// Elementwise activation of node `input`.
    Activation {
        input: usize,
        activation: Activation,
        width: usize,
    },
    /// `y[j] = x[indices[j]]` over node `input`.
    Gather { input: usize, indices: &'a [u32] },
    /// Elementwise `a + b`.
    Add { a: usize, b: usize, width: usize },
    /// Elementwise `a * b`.
    Mul { a: usize, b: usize, width: usize },
    /// `inputs` laid end to end.
    Concat { inputs: &'a [u32], width: usize },
}

fn malformed(node: usize, what: &'static str) -> PredictError {
    PredictError::GraphMalformed { node, what }
}

fn require_empty(node: usize, s: Section, what: &'static str) -> Result<(), PredictError> {
    if s.is_empty() {
        Ok(())
    } else {
        Err(malformed(node, what))
    }
}

fn add_weights(total: &mut usize, n: usize) -> Result<(), PredictError> {
    *total = total
        .checked_add(n)
        .filter(|&t| t <= MAX_TOTAL_WEIGHTS)
        .ok_or(PredictError::DimensionOverflow {
            what: "total weights (limits::MAX_TOTAL_WEIGHTS)",
        })?;
    Ok(())
}

/// Validate a Dense node's weight / scale / bias sections against
/// `in_dim × out_dim` and its dtype. Shared by v4 parsing; v3 layers run
/// the equivalent checks in `model.rs` (kept there so v3 errors are
/// byte-for-byte what they were).
fn validate_dense_sections(
    node: usize,
    l: &LayerOffsets,
    bytes: &[u8],
) -> Result<(), PredictError> {
    let in_dim = l.in_dim as usize;
    let out_dim = l.out_dim as usize;
    let n_weights = in_dim
        .checked_mul(out_dim)
        .ok_or(PredictError::DimensionOverflow {
            what: "node.in_dim * node.out_dim",
        })?;
    match l.weight_dtype {
        WeightDtype::F32 => {
            cast_f32_section("node.weights[f32]", l.weights, bytes, n_weights)?;
        }
        WeightDtype::F16 => {
            cast_u16_section("node.weights[f16]", l.weights, bytes, n_weights)?;
        }
        WeightDtype::I8 => {
            cast_i8_section("node.weights[i8]", l.weights, bytes, n_weights)?;
            cast_f32_section("node.scales", l.scales, bytes, out_dim)?;
        }
    }
    if !matches!(l.weight_dtype, WeightDtype::I8) {
        require_empty(node, l.scales, "scales section is only valid for I8 Dense")?;
    }
    if !l.biases.is_empty() {
        cast_f32_section("node.biases", l.biases, bytes, out_dim)?;
    }
    Ok(())
}

/// Parse and validate a v4 node table. `bytes` is the owned,
/// decompressed bake.
pub(crate) fn parse_v4(
    bytes: &[u8],
    node_table: Section,
    n_nodes: usize,
    n_inputs: usize,
    n_outputs: usize,
) -> Result<Graph, PredictError> {
    if n_nodes < 2 {
        return Err(malformed(
            0,
            "a graph needs an Input node and at least one op after it",
        ));
    }
    if n_nodes > MAX_NODES {
        return Err(PredictError::DimensionOverflow {
            what: "n_nodes (limits::MAX_NODES)",
        });
    }
    let raw = node_table.slice("node_table", bytes)?;
    let expected = n_nodes
        .checked_mul(NODE_ENTRY_SIZE)
        .ok_or(PredictError::DimensionOverflow {
            what: "n_nodes * NODE_ENTRY_SIZE",
        })?;
    if raw.len() != expected {
        return Err(PredictError::SectionOutOfRange {
            what: "node_table",
            offset: node_table.offset,
            len: node_table.len,
            file_len: bytes.len(),
        });
    }
    let entries: &[NodeEntry] =
        bytemuck::try_cast_slice(raw).map_err(|_| PredictError::SectionMisaligned {
            what: "node_table",
            offset: node_table.offset,
            required_align: core::mem::align_of::<NodeEntry>(),
        })?;

    let mut nodes: Vec<GraphNode> = try_vec_with_capacity(n_nodes)?;
    let mut dense_nodes: Vec<u32> = try_vec_with_capacity(n_nodes)?;
    let mut total_weights = 0usize;

    for (i, e) in entries.iter().enumerate() {
        let op = e.op;
        if !matches!(
            op,
            OP_INPUT | OP_DENSE | OP_ACTIVATION | OP_GATHER | OP_ADD | OP_MUL | OP_CONCAT
        ) {
            return Err(PredictError::UnknownGraphOp { node: i, byte: op });
        }
        if e.flags != 0 || e.reserved != [0, 0] {
            return Err(malformed(i, "flags and reserved fields must be zero"));
        }
        let width = e.out_dim as usize;
        if width == 0 {
            return Err(PredictError::ZeroDimension {
                what: "node.out_dim",
            });
        }
        if width > MAX_DIM {
            return Err(PredictError::DimensionOverflow {
                what: "node.out_dim",
            });
        }
        if (i == 0) != (op == OP_INPUT) {
            return Err(malformed(i, "node 0 must be the only Input node"));
        }

        // Arity, checked against the op before the indices are read.
        if !(e.inputs.len as usize).is_multiple_of(4) {
            return Err(malformed(i, "inputs section length is not a multiple of 4"));
        }
        let arity = e.inputs.len as usize / 4;
        let arity_ok = match op {
            OP_INPUT => arity == 0,
            OP_DENSE | OP_ACTIVATION | OP_GATHER => arity == 1,
            OP_ADD | OP_MUL => arity == 2,
            _ => (1..=MAX_NODE_INPUTS).contains(&arity),
        };
        if !arity_ok {
            return Err(malformed(i, "wrong number of inputs for op"));
        }
        let ins: &[u32] = cast_u32_section("node.inputs", e.inputs, bytes, arity)?;
        for &inp in ins {
            if inp as usize >= i {
                return Err(PredictError::GraphInputRef {
                    node: i,
                    input: inp as usize,
                });
            }
        }
        let width_of = |j: u32| nodes[j as usize].width as usize;

        // Fields an op doesn't use must be zero / empty.
        if op != OP_DENSE {
            if e.weight_dtype != 0 {
                return Err(malformed(i, "weight_dtype is only valid for Dense"));
            }
            require_empty(i, e.data1, "data1 is only valid for Dense")?;
            require_empty(i, e.data2, "data2 is only valid for Dense")?;
            if op != OP_GATHER {
                require_empty(i, e.data0, "data0 is only valid for Dense and Gather")?;
            }
            if op != OP_ACTIVATION && e.activation != 0 {
                return Err(malformed(
                    i,
                    "activation is only valid for Dense and Activation",
                ));
            }
        }

        let kind = match op {
            OP_INPUT => {
                if width != n_inputs {
                    return Err(PredictError::GraphShapeMismatch {
                        node: i,
                        expected: n_inputs,
                        got: width,
                    });
                }
                NodeKind::Input
            }
            OP_DENSE => {
                let input = ins[0];
                let layer = LayerOffsets {
                    in_dim: width_of(input) as u32,
                    out_dim: e.out_dim,
                    activation: Activation::from_byte(e.activation)?,
                    weight_dtype: WeightDtype::from_byte(e.weight_dtype)?,
                    weights: e.data0,
                    scales: e.data1,
                    biases: e.data2,
                };
                validate_dense_sections(i, &layer, bytes)?;
                add_weights(
                    &mut total_weights,
                    layer.in_dim as usize * layer.out_dim as usize,
                )?;
                dense_nodes.push(i as u32);
                NodeKind::Dense { input, layer }
            }
            OP_ACTIVATION => {
                let input = ins[0];
                if width_of(input) != width {
                    return Err(PredictError::GraphShapeMismatch {
                        node: i,
                        expected: width_of(input),
                        got: width,
                    });
                }
                NodeKind::Activation {
                    input,
                    activation: Activation::from_byte(e.activation)?,
                }
            }
            OP_GATHER => {
                let input = ins[0];
                let src_width = width_of(input);
                let idx = cast_u32_section("node.gather_indices", e.data0, bytes, width)?;
                for &k in idx {
                    if k as usize >= src_width {
                        return Err(malformed(i, "gather index out of range of its input"));
                    }
                }
                let start = idx[0];
                let contiguous = idx
                    .iter()
                    .enumerate()
                    .all(|(j, &k)| k as usize == start as usize + j);
                NodeKind::Gather {
                    input,
                    indices: e.data0,
                    contiguous_start: contiguous.then_some(start),
                }
            }
            OP_ADD | OP_MUL => {
                let (a, b) = (ins[0], ins[1]);
                for src in [a, b] {
                    if width_of(src) != width {
                        return Err(PredictError::GraphShapeMismatch {
                            node: i,
                            expected: width,
                            got: width_of(src),
                        });
                    }
                }
                if op == OP_ADD {
                    NodeKind::Add { a, b }
                } else {
                    NodeKind::Mul { a, b }
                }
            }
            _ => {
                // OP_CONCAT (op byte was checked above).
                let mut sum = 0usize;
                for &src in ins {
                    sum =
                        sum.checked_add(width_of(src))
                            .ok_or(PredictError::DimensionOverflow {
                                what: "concat width",
                            })?;
                }
                if sum != width {
                    return Err(PredictError::GraphShapeMismatch {
                        node: i,
                        expected: sum,
                        got: width,
                    });
                }
                NodeKind::Concat { inputs: e.inputs }
            }
        };
        nodes.push(GraphNode {
            kind,
            width: e.out_dim,
            slot: 0,
        });
    }

    let last = nodes.len() - 1;
    let out_width = nodes[last].width as usize;
    if out_width != n_outputs {
        return Err(PredictError::OutputDimMismatch {
            expected: n_outputs,
            got: out_width,
        });
    }

    let is_chain =
        nodes.iter().enumerate().skip(1).all(
            |(i, n)| matches!(n.kind, NodeKind::Dense { input, .. } if input as usize == i - 1),
        );
    let arena_len = plan_slots(&mut nodes, bytes)?;
    Ok(Graph {
        nodes,
        arena_len,
        dense_nodes,
        is_chain,
    })
}

/// Lower a validated v3 layer chain to `Input → Dense → … → Dense`.
pub(crate) fn lower_chain(
    layers: &[LayerOffsets],
    n_inputs: usize,
    bytes: &[u8],
) -> Result<Graph, PredictError> {
    let n_nodes = layers.len() + 1;
    let mut nodes: Vec<GraphNode> = try_vec_with_capacity(n_nodes)?;
    let mut dense_nodes: Vec<u32> = try_vec_with_capacity(layers.len())?;
    nodes.push(GraphNode {
        kind: NodeKind::Input,
        width: n_inputs as u32,
        slot: 0,
    });
    let mut total_weights = 0usize;
    for (k, layer) in layers.iter().enumerate() {
        add_weights(
            &mut total_weights,
            layer.in_dim as usize * layer.out_dim as usize,
        )?;
        dense_nodes.push(k as u32 + 1);
        nodes.push(GraphNode {
            kind: NodeKind::Dense {
                input: k as u32,
                layer: *layer,
            },
            width: layer.out_dim,
            slot: 0,
        });
    }
    let arena_len = plan_slots(&mut nodes, bytes)?;
    Ok(Graph {
        nodes,
        arena_len,
        dense_nodes,
        is_chain: true,
    })
}

/// Call `f` once per input of `kind`, in order (repeats included).
pub(crate) fn for_each_input(kind: &NodeKind, bytes: &[u8], mut f: impl FnMut(usize)) {
    match *kind {
        NodeKind::Input => {}
        NodeKind::Dense { input, .. }
        | NodeKind::Activation { input, .. }
        | NodeKind::Gather { input, .. } => f(input as usize),
        NodeKind::Add { a, b } | NodeKind::Mul { a, b } => {
            f(a as usize);
            f(b as usize);
        }
        NodeKind::Concat { inputs } => {
            for &j in concat_inputs(inputs, bytes) {
                f(j as usize);
            }
        }
    }
}

/// The validated u32 index list of a Concat node.
pub(crate) fn concat_inputs(inputs: Section, bytes: &[u8]) -> &[u32] {
    let n = inputs.len as usize / 4;
    cast_u32_section("node.inputs", inputs, bytes, n).expect("concat inputs validated at load")
}

/// The validated u32 index list of a Gather node.
pub(crate) fn gather_indices(indices: Section, width: usize, bytes: &[u8]) -> &[u32] {
    cast_u32_section("node.gather_indices", indices, bytes, width)
        .expect("gather indices validated at load")
}

/// Assign every node except the last a scratch slot, reusing space once
/// a node's last reader has run; reject dead nodes; return the arena
/// length.
///
/// First-fit over a sorted free list. A node's slot is allocated before
/// its dying inputs are released, so a node never writes over a buffer
/// it reads. Deterministic: the same graph always gets the same slots.
fn plan_slots(nodes: &mut [GraphNode], bytes: &[u8]) -> Result<usize, PredictError> {
    let n = nodes.len();
    let last = n - 1;
    // last_use[j] = index of the last node reading j (usize::MAX = none).
    let mut last_use: Vec<usize> = try_vec_with_capacity(n)?;
    last_use.resize(n, usize::MAX);
    for (i, node) in nodes.iter().enumerate() {
        for_each_input(&node.kind, bytes, |j| last_use[j] = i);
    }
    for (j, &u) in last_use.iter().enumerate().take(last) {
        if u == usize::MAX {
            return Err(malformed(j, "dead node: its output is never read"));
        }
    }

    // Sorted, non-adjacent free intervals (start, len).
    let mut free: Vec<(usize, usize)> = try_vec_with_capacity(n)?;
    let mut high_water = 0usize;
    let mut dying: Vec<usize> = try_vec_with_capacity(MAX_NODE_INPUTS)?;
    for i in 0..n {
        if i != last {
            let w = nodes[i].width as usize;
            let slot = match free.iter().position(|&(_, len)| len >= w) {
                Some(k) => {
                    let (start, len) = free[k];
                    if len == w {
                        free.remove(k);
                    } else {
                        free[k] = (start + w, len - w);
                    }
                    start
                }
                None => {
                    let start = high_water;
                    high_water = high_water
                        .checked_add(w)
                        .filter(|&h| h <= MAX_SCRATCH_ELEMS)
                        .ok_or(PredictError::DimensionOverflow {
                            what: "scratch arena (limits::MAX_SCRATCH_ELEMS)",
                        })?;
                    start
                }
            };
            nodes[i].slot = slot as u32;
        }
        dying.clear();
        for_each_input(&nodes[i].kind, bytes, |j| {
            if last_use[j] == i && !dying.contains(&j) {
                dying.push(j);
            }
        });
        for &j in &dying {
            release(&mut free, nodes[j].slot as usize, nodes[j].width as usize);
        }
    }
    Ok(high_water)
}

fn release(free: &mut Vec<(usize, usize)>, start: usize, len: usize) {
    let k = free.partition_point(|&(s, _)| s < start);
    free.insert(k, (start, len));
    // Merge with the next interval, then with the previous one.
    if k + 1 < free.len() && free[k].0 + free[k].1 == free[k + 1].0 {
        free[k].1 += free[k + 1].1;
        free.remove(k + 1);
    }
    if k > 0 && free[k - 1].0 + free[k - 1].1 == free[k].0 {
        free[k - 1].1 += free[k].1;
        free.remove(k);
    }
}
