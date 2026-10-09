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

use bytemuck::Zeroable;
use zenpredict::wire::{
    HEADER_SIZE, NODE_ENTRY_SIZE, OP_ACTIVATION, OP_ADD, OP_CONCAT, OP_DENSE, OP_GATHER, OP_INPUT,
    OP_MUL, SECTION_OFF_LAYER_TABLE, SECTION_OFF_SCALER_MEAN, SECTION_OFF_SCALER_SCALE,
};
use zenpredict::{
    Activation, FeatureBound, NodeEntry, OutputSpec, Section, SparseOverride, WeightDtype,
};

use crate::composer::{
    BakeError, BakeLayer, BakeMetadataEntry, BakeRequest, TailSections, append_f32,
    append_layer_weights, compress_payload, pad_to, validate_common_post, validate_common_pre,
    write_section, write_tail_sections,
};

/// One node of a v4 graph bake. Input indices (`input`, `a`, `b`,
/// `inputs`) name earlier nodes.
///
/// Build nodes with the constructors ([`Self::input`], [`Self::dense`],
/// …). Every variant is `#[non_exhaustive]`, so fields can be added later
/// without breaking callers; match with `..`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum BakeNode<'a> {
    /// The scaled feature vector. Must be node 0, and the only one.
    /// `width` is the model's `n_inputs`.
    #[non_exhaustive]
    Input { width: usize },
    /// `activation(biases + x·W)` over node `input`. `weights` is
    /// row-major `in_dim × out_dim` (`in_dim` = `input`'s width), stored
    /// as `dtype` exactly as a v3 layer. `biases: None` omits the bias
    /// (computes as all `+0.0`).
    #[non_exhaustive]
    Dense {
        input: u32,
        out_dim: usize,
        activation: Activation,
        dtype: WeightDtype,
        weights: &'a [f32],
        biases: Option<&'a [f32]>,
    },
    /// Elementwise `activation` of node `input`.
    #[non_exhaustive]
    Activation { input: u32, activation: Activation },
    /// `y[j] = x[indices[j]]` over node `input`.
    #[non_exhaustive]
    Gather { input: u32, indices: &'a [u32] },
    /// Elementwise `a + b` (equal widths).
    #[non_exhaustive]
    Add { a: u32, b: u32 },
    /// Elementwise `a * b` (equal widths).
    #[non_exhaustive]
    Mul { a: u32, b: u32 },
    /// `inputs` laid end to end.
    #[non_exhaustive]
    Concat { inputs: &'a [u32] },
}

impl<'a> BakeNode<'a> {
    /// The scaled feature vector (node 0); `width` = `n_inputs`.
    pub const fn input(width: usize) -> Self {
        Self::Input { width }
    }

    /// `activation(biases + x·W)` over node `input`; see [`Self::Dense`].
    pub const fn dense(
        input: u32,
        out_dim: usize,
        activation: Activation,
        dtype: WeightDtype,
        weights: &'a [f32],
        biases: Option<&'a [f32]>,
    ) -> Self {
        Self::Dense {
            input,
            out_dim,
            activation,
            dtype,
            weights,
            biases,
        }
    }

    /// Elementwise `activation` of node `input`.
    pub const fn activation(input: u32, activation: Activation) -> Self {
        Self::Activation { input, activation }
    }

    /// `y[j] = x[indices[j]]` over node `input`.
    pub const fn gather(input: u32, indices: &'a [u32]) -> Self {
        Self::Gather { input, indices }
    }

    /// Elementwise `a + b`.
    pub const fn add(a: u32, b: u32) -> Self {
        Self::Add { a, b }
    }

    /// Elementwise `a * b`.
    pub const fn mul(a: u32, b: u32) -> Self {
        Self::Mul { a, b }
    }

    /// `inputs` laid end to end.
    pub const fn concat(inputs: &'a [u32]) -> Self {
        Self::Concat { inputs }
    }
}

/// Everything a v4 graph bake needs: the node list plus the sections
/// outside the network (scaler, metadata, output specs, …), each with the
/// meaning it has in [`BakeRequest`]. No layer-chain fields: no `layers`,
/// no feature/output permutations or hidden-unit reorder.
///
/// Build with [`Self::new`] and the chained setters; bake with
/// [`bake_graph`] or [`Self::bake`].
#[non_exhaustive]
#[derive(Clone, Copy)]
pub struct GraphBakeRequest<'a> {
    pub schema_hash: u64,
    /// Header flags. Bits 0..=3 are managed by the composer
    /// (compression); bits 4..=15 are reserved. Leave 0.
    pub flags: u16,
    pub scaler_mean: &'a [f32],
    pub scaler_scale: &'a [f32],
    pub nodes: &'a [BakeNode<'a>],
    pub feature_bounds: &'a [FeatureBound],
    pub metadata: &'a [BakeMetadataEntry<'a>],
    pub output_specs: &'a [OutputSpec],
    pub discrete_sets: &'a [f32],
    pub sparse_overrides: &'a [SparseOverride],
    /// LZ4-compress the payload (loader decompresses transparently).
    pub compressed: bool,
}

impl<'a> GraphBakeRequest<'a> {
    /// The required fields; every optional section empty, uncompressed.
    pub const fn new(
        schema_hash: u64,
        scaler_mean: &'a [f32],
        scaler_scale: &'a [f32],
        nodes: &'a [BakeNode<'a>],
    ) -> Self {
        Self {
            schema_hash,
            flags: 0,
            scaler_mean,
            scaler_scale,
            nodes,
            feature_bounds: &[],
            metadata: &[],
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            compressed: false,
        }
    }

    /// Header flags (bits 4..=15 reserved).
    pub const fn flags(mut self, flags: u16) -> Self {
        self.flags = flags;
        self
    }

    /// Per-input `[low, high]` pairs (empty = absent).
    pub const fn feature_bounds(mut self, bounds: &'a [FeatureBound]) -> Self {
        self.feature_bounds = bounds;
        self
    }

    /// Typed-TLV metadata entries.
    pub const fn metadata(mut self, entries: &'a [BakeMetadataEntry<'a>]) -> Self {
        self.metadata = entries;
        self
    }

    /// Per-output specs (length `n_outputs`, or empty).
    pub const fn output_specs(mut self, specs: &'a [OutputSpec]) -> Self {
        self.output_specs = specs;
        self
    }

    /// f32 pool the output specs' discrete sets slice into.
    pub const fn discrete_sets(mut self, pool: &'a [f32]) -> Self {
        self.discrete_sets = pool;
        self
    }

    /// Sparse `(idx, value)` overrides applied after the output specs.
    pub const fn sparse_overrides(mut self, overrides: &'a [SparseOverride]) -> Self {
        self.sparse_overrides = overrides;
        self
    }

    /// Whole-payload LZ4 compression.
    pub const fn compressed(mut self, enabled: bool) -> Self {
        self.compressed = enabled;
        self
    }

    /// Equivalent to [`bake_graph`]`(&self)`.
    pub fn bake(&self) -> Result<Vec<u8>, BakeError> {
        bake_graph(self)
    }

    /// The shared-section view, for the validators and writers `bake`
    /// uses (no layers, no permutations).
    fn sections(&self) -> BakeRequest<'a> {
        let mut r = BakeRequest::new(
            self.schema_hash,
            self.flags,
            self.scaler_mean,
            self.scaler_scale,
            &[],
        );
        r.feature_bounds = self.feature_bounds;
        r.metadata = self.metadata;
        r.output_specs = self.output_specs;
        r.discrete_sets = self.discrete_sets;
        r.sparse_overrides = self.sparse_overrides;
        r.compressed = self.compressed;
        r
    }
}

fn invalid(node: usize, what: &'static str) -> BakeError {
    BakeError::GraphInvalid { node, what }
}

/// Compose a ZNPR v4 graph bake.
///
/// No hidden-unit reorder or zero-bias pass runs on graphs (the JSON
/// baker applies `zerobias_tau` to Dense weights before calling this).
///
/// The output loads with [`zenpredict::Model::from_bytes`]; this function
/// loads it once itself before returning, so a graph zenpredict would
/// reject is a [`BakeError::GraphRejected`] here, never a file on disk.
pub fn bake_graph(graph: &GraphBakeRequest<'_>) -> Result<Vec<u8>, BakeError> {
    let nodes = graph.nodes;
    let req = &graph.sections();
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
        let mut entry = NodeEntry::zeroed();
        entry.op = op;
        entry.activation = activation;
        entry.weight_dtype = dtype;
        entry.out_dim = widths[i] as u32;
        entry.inputs = inputs_section;
        entry.data0 = data0;
        entry.data1 = data1;
        entry.data2 = data2;
        // flags and reserved stay zero. The wire format is little-endian;
        // like the rest of the crate this assumes an LE host.
        let off = HEADER_SIZE + i * NODE_ENTRY_SIZE;
        buf[off..off + NODE_ENTRY_SIZE].copy_from_slice(bytemuck::bytes_of(&entry));
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
