//! ZNPR v4 graph composer.
//!
//! [`bake_graph`] writes a static op graph (see zenpredict's
//! `docs/ZNPR_V4_GRAPH.md`) around the same scaler / metadata / output-spec
//! sections [`crate::bake`] writes for a v3 chain. Nodes are given in
//! topological order: every input index names an earlier node, node 0 is
//! the only [`BakeNode::Input`], and the last node is the model output.
//!
//! The composer checks only what it needs to lay bytes out (input
//! references, payload lengths); the finished bake is then loaded with
//! [`zenpredict::Model::from_bytes`], which owns every graph rule, and a
//! rejection comes back as [`BakeError::GraphRejected`].

use alloc::vec::Vec;

use zenpredict::wire::{
    HEADER_SIZE, NODE_ENTRY_SIZE, NODE_OFF_DATA0, NODE_OFF_DATA1, NODE_OFF_DATA2, NODE_OFF_INPUTS,
    OP_ACTIVATION, OP_ADD, OP_CONCAT, OP_DENSE, OP_GATHER, OP_INPUT, OP_MUL,
    SECTION_OFF_LAYER_TABLE, SECTION_OFF_SCALER_MEAN, SECTION_OFF_SCALER_SCALE,
};
use zenpredict::{Activation, Section, WeightDtype};

use crate::composer::{
    BakeError, BakeLayer, BakeRequest, TailSections, append_f32, append_layer_weights,
    compress_payload, pad_to, validate_common_post, validate_common_pre, write_section,
    write_section_inline, write_tail_sections,
};

/// One node of a v4 graph bake. Input indices (`input`, `a`, `b`,
/// `inputs`) name earlier nodes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum BakeNode<'a> {
    /// The scaled feature vector. Must be node 0, and the only one.
    /// `width` is the model's `n_inputs`.
    Input { width: usize },
    /// `activation(biases + x·W)` over node `input`. `weights` is
    /// row-major `in_dim × out_dim` (`in_dim` = `input`'s width), stored
    /// as `dtype` exactly as a v3 layer. `biases: None` omits the bias
    /// (computes as all `+0.0`).
    Dense {
        input: u32,
        out_dim: usize,
        activation: Activation,
        dtype: WeightDtype,
        weights: &'a [f32],
        biases: Option<&'a [f32]>,
    },
    /// Elementwise `activation` of node `input`.
    Activation { input: u32, activation: Activation },
    /// `y[j] = x[indices[j]]` over node `input`.
    Gather { input: u32, indices: &'a [u32] },
    /// Elementwise `a + b` (equal widths).
    Add { a: u32, b: u32 },
    /// Elementwise `a * b` (equal widths).
    Mul { a: u32, b: u32 },
    /// `inputs` laid end to end.
    Concat { inputs: &'a [u32] },
}

fn invalid(node: usize, what: &'static str) -> BakeError {
    BakeError::GraphInvalid { node, what }
}

/// Compose a ZNPR v4 graph bake.
///
/// `req` supplies everything outside the network — schema hash, flags,
/// scaler, feature bounds, metadata, output specs, discrete sets, sparse
/// overrides, compression — with the meaning it has for [`crate::bake`].
/// `req.layers` must be empty, and `feature_order`, `output_order` and
/// `hu_permutations` must be `None`: those reorderings are defined on a
/// layer chain. No hidden-unit reorder or zero-bias pass runs on graphs.
///
/// The output loads with [`zenpredict::Model::from_bytes`]; this function
/// loads it once itself before returning, so a graph zenpredict would
/// reject is a [`BakeError::GraphRejected`] here, never a file on disk.
pub fn bake_graph(req: &BakeRequest<'_>, nodes: &[BakeNode<'_>]) -> Result<Vec<u8>, BakeError> {
    if !req.layers.is_empty() {
        return Err(invalid(
            0,
            "BakeRequest.layers must be empty for a graph bake",
        ));
    }
    if req.feature_order.is_some() || req.output_order.is_some() || req.hu_permutations.is_some() {
        return Err(invalid(
            0,
            "feature_order, output_order and hu_permutations apply to layer chains only",
        ));
    }
    let n_inputs = match nodes.first() {
        Some(BakeNode::Input { width }) => *width,
        _ => return Err(invalid(0, "node 0 must be Input")),
    };

    // Widths, and the minimal checks needed to lay the payload out.
    let mut widths: Vec<usize> = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let w = |j: u32| -> Result<usize, BakeError> {
            if (j as usize) < i {
                Ok(widths[j as usize])
            } else {
                Err(invalid(i, "input must reference an earlier node"))
            }
        };
        let width = match *node {
            BakeNode::Input { width } => {
                if i != 0 {
                    return Err(invalid(i, "only node 0 may be Input"));
                }
                width
            }
            BakeNode::Dense {
                input,
                out_dim,
                weights,
                biases,
                ..
            } => {
                let in_dim = w(input)?;
                if Some(weights.len()) != in_dim.checked_mul(out_dim) {
                    return Err(invalid(i, "Dense weights length != in_dim * out_dim"));
                }
                if biases.is_some_and(|b| b.len() != out_dim) {
                    return Err(invalid(i, "Dense biases length != out_dim"));
                }
                out_dim
            }
            BakeNode::Activation { input, .. } => w(input)?,
            BakeNode::Gather { input, indices } => {
                w(input)?;
                indices.len()
            }
            BakeNode::Add { a, b } | BakeNode::Mul { a, b } => {
                w(b)?;
                w(a)?
            }
            BakeNode::Concat { inputs } => {
                let mut sum = 0usize;
                for &j in inputs {
                    sum = sum
                        .checked_add(w(j)?)
                        .ok_or(invalid(i, "Concat width overflows"))?;
                }
                sum
            }
        };
        widths.push(width);
    }
    let n_outputs = *widths.last().expect("nodes is non-empty");
    let n_nodes = nodes.len();

    validate_common_pre(req, n_inputs, n_outputs)?;
    validate_common_post(req, n_inputs)?;

    let table_len = n_nodes
        .checked_mul(NODE_ENTRY_SIZE)
        .ok_or(invalid(0, "node table size overflows"))?;
    let mut buf = Vec::with_capacity(HEADER_SIZE + table_len + 4096);
    buf.resize(HEADER_SIZE + table_len, 0);

    buf[0..4].copy_from_slice(b"ZNPR");
    buf[4..6].copy_from_slice(&zenpredict::GRAPH_FORMAT_VERSION.to_le_bytes());
    buf[6..8].copy_from_slice(&req.flags.to_le_bytes());
    buf[8..12].copy_from_slice(&(n_inputs as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(n_outputs as u32).to_le_bytes());
    // v4 reuses the v3 `n_layers` field for the node count …
    buf[16..20].copy_from_slice(&(n_nodes as u32).to_le_bytes());
    buf[24..32].copy_from_slice(&req.schema_hash.to_le_bytes());
    // … and the `layer_table` Section for the node table.
    write_section(
        &mut buf,
        SECTION_OFF_LAYER_TABLE,
        Section::new(HEADER_SIZE as u32, table_len as u32),
    );

    pad_to(&mut buf, 4);
    let mean = append_f32(&mut buf, req.scaler_mean);
    write_section(&mut buf, SECTION_OFF_SCALER_MEAN, mean);
    pad_to(&mut buf, 4);
    let scale = append_f32(&mut buf, req.scaler_scale);
    write_section(&mut buf, SECTION_OFF_SCALER_SCALE, scale);

    for (i, node) in nodes.iter().enumerate() {
        let (op, activation, dtype, input_list): (u8, u8, u8, &[u32]) = match node {
            BakeNode::Input { .. } => (OP_INPUT, 0, 0, &[]),
            BakeNode::Dense {
                input,
                activation,
                dtype,
                ..
            } => (
                OP_DENSE,
                *activation as u8,
                *dtype as u8,
                core::slice::from_ref(input),
            ),
            BakeNode::Activation { input, activation } => (
                OP_ACTIVATION,
                *activation as u8,
                0,
                core::slice::from_ref(input),
            ),
            BakeNode::Gather { input, .. } => (OP_GATHER, 0, 0, core::slice::from_ref(input)),
            BakeNode::Add { a, b } => (OP_ADD, 0, 0, &[*a, *b][..]),
            BakeNode::Mul { a, b } => (OP_MUL, 0, 0, &[*a, *b][..]),
            BakeNode::Concat { inputs } => (OP_CONCAT, 0, 0, *inputs),
        };
        let inputs_section = append_u32(&mut buf, input_list);
        let (data0, data1, data2) = match *node {
            BakeNode::Dense {
                out_dim,
                activation,
                dtype,
                weights,
                biases,
                ..
            } => {
                let layer = BakeLayer {
                    in_dim: widths[input_list[0] as usize],
                    out_dim,
                    activation,
                    dtype,
                    weights,
                    biases: biases.unwrap_or(&[]),
                };
                let (w, s) = append_layer_weights(&mut buf, &layer);
                let b = match biases {
                    Some(b) => {
                        pad_to(&mut buf, 4);
                        append_f32(&mut buf, b)
                    }
                    None => Section::empty(),
                };
                (w, s, b)
            }
            BakeNode::Gather { indices, .. } => (
                append_u32(&mut buf, indices),
                Section::empty(),
                Section::empty(),
            ),
            _ => (Section::empty(), Section::empty(), Section::empty()),
        };
        let off = HEADER_SIZE + i * NODE_ENTRY_SIZE;
        let entry = &mut buf[off..off + NODE_ENTRY_SIZE];
        entry[0] = op;
        entry[1] = activation;
        entry[2] = dtype;
        // [3] flags = 0.
        entry[4..8].copy_from_slice(&(widths[i] as u32).to_le_bytes());
        write_section_inline(entry, NODE_OFF_INPUTS, inputs_section);
        write_section_inline(entry, NODE_OFF_DATA0, data0);
        write_section_inline(entry, NODE_OFF_DATA1, data1);
        write_section_inline(entry, NODE_OFF_DATA2, data2);
        // [40..48] reserved = 0.
    }

    let TailSections { .. } = write_tail_sections(&mut buf, req);
    if req.compressed {
        compress_payload(&mut buf);
    }

    zenpredict::Model::from_bytes(&buf).map_err(BakeError::GraphRejected)?;
    Ok(buf)
}

fn append_u32(buf: &mut Vec<u8>, values: &[u32]) -> Section {
    if values.is_empty() {
        return Section::empty();
    }
    pad_to(buf, 4);
    let start = buf.len() as u32;
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    Section::new(start, (values.len() * 4) as u32)
}
