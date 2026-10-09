# ZNPR v4 — static op graphs

**Status:** design, 2026-10-09. Implementation lands in the same PR. Unpublished
(zenpredict 0.2.0 is not on crates.io yet; 0.1.0 there is ZNPR v2 only).

## Why

ZNPR v3 runs one fixed shape: `scaler → [Dense + bias + activation] × N`. Every new
head shape (a gated head, a two-branch head, a mixture) needs new runtime code in
zenpredict before a trainer can ship it. zensim's next candidate (E33 arm B) is

```text
g = Σ_k v_k · ReLU(w_k · d) · exp(u_k · r)
```

where `d` are difference features and `r` reference-only features — two branches
multiplied elementwise, then summed. That is not a layer chain, and it is not a new
*layer type* either: it is four ordinary ops (Gather, Dense, Exp, Mul) wired as a
small graph.

v4 replaces only the layer chain with a static, topologically ordered op graph, so
new heads become trainer/export work. Everything else in the file keeps its v3
meaning: header, scaler, feature transforms (metadata), feature bounds, metadata
TLV, output specs, discrete sets, sparse overrides, compression.

## Format

### Header

The 128-byte v3 `Header` is reused byte for byte, with three differences:

| bytes  | v3 meaning                     | v4 meaning                               |
|--------|--------------------------------|------------------------------------------|
| 4..6   | `version = 3`                  | `version = 4`                            |
| 16..20 | `n_layers`                     | `n_nodes`                                |
| 48..56 | `layer_table` (LayerEntry[])   | `node_table` (NodeEntry[n_nodes])        |

`feature_order` / `output_order` (100..116) must be empty in v4: the input/output
permutations are defined against `layer[0]` rows and `layer[last]` columns, which a
graph does not have. A non-empty section is rejected at load. (The bake-side
optimizer that produces them only runs on chains.)

Whole-payload LZ4 compression (flags bit 0 + algo nibble) is format-agnostic and
works unchanged.

### NodeEntry (48 bytes, little-endian, `#[repr(C)]`)

```text
0       op: u8            0=Input 1=Dense 2=Activation 3=Gather 4=Add 5=Mul 6=Concat
1       activation: u8    Dense: fused activation after bias. Activation: the function. Else 0.
2       weight_dtype: u8  Dense: 0=F32 1=F16 2=I8. Else 0.
3       flags: u8         reserved, must be 0
4..8    out_dim: u32      width of this node's output vector
8..16   inputs: Section   u32 LE node indices, `arity * 4` bytes
16..24  data0: Section    Dense: weights. Gather: u32 LE source indices (out_dim of them).
24..32  data1: Section    Dense I8: per-output f32 scales. Else empty.
32..40  data2: Section    Dense: f32 biases (out_dim), or empty = no bias. Else empty.
40..48  reserved: [u32; 2], must be 0
```

Sections are the v3 `(offset: u32, len: u32)` pair into the (decompressed) file.
Alignment rules are v3's: f32/u32 payloads 4-aligned, f16 2-aligned.

Dense weights use the v3 layout exactly: row-major, input-major,
`W[i * out_dim + o]`; F16 is raw binary16 bits; I8 is `q[i, o] * scales[o]`.

### Ops

| op         | arity | out_dim must equal                  | computes |
|------------|-------|-------------------------------------|----------|
| Input      | 0     | header `n_inputs`                   | the scaled feature vector `(x - mean) / scale` (v3 scaler, unchanged) |
| Dense      | 1     | any (1..=MAX_DIM)                   | `act(b + x·W)`, v3 kernel and dequant unchanged |
| Activation | 1     | width(input)                        | `act(x)` elementwise |
| Gather     | 1     | len(indices)                        | `y[j] = x[indices[j]]`; a slice is a Gather of a contiguous range |
| Add        | 2     | width(a) = width(b)                 | `a + b` elementwise |
| Mul        | 2     | width(a) = width(b)                 | `a * b` elementwise |
| Concat     | 1..=MAX_NODE_INPUTS | Σ width(inputs)       | inputs laid end to end, in order |

No broadcasting. A width mismatch is a load error, not a runtime one.

### Activations

| byte | name      | definition |
|------|-----------|------------|
| 0    | Identity  | `x` |
| 1    | ReLU      | `if x < 0 { 0 } else { x }` (keeps NaN and -0.0, as v3) |
| 2    | LeakyReLU | `if x < 0 { 0.01 * x } else { x }` (as v3) |
| 3    | Exp       | `expf(clamp(x, -30, 30))` |
| 4    | Softplus  | `if x > 20 { x } else { log1pf(expf(x)) }` |

Bytes 3 and 4 are v4-only: a v3 `LayerEntry` carrying them still fails with
`UnknownActivation`, exactly as today.

`Exp` clamps its input to `[-30, 30]` (the same bound `ScoreTransform::Exp` uses),
so it never returns ±inf for finite or infinite input. That matters for gated
heads: `ReLU(0) · exp(huge)` must be `0`, not `0 · inf = NaN`. `Softplus` matches
PyTorch `nn.Softplus(beta=1, threshold=20)`. Both call `libm::expf` /
`libm::log1pf` on **every** build (std and no_std), so they return the same bits on
every platform; a platform `expf` would not. They run scalar: vectorizing them would
need a polynomial that a trainer mirrors bit for bit, which is a separate decision.
The trainer must apply the same clamp and threshold (constants
`zenpredict::EXP_INPUT_CLAMP` and `zenpredict::SOFTPLUS_THRESHOLD`).

### Graph rules (validated at load, before any compute)

1. `2 <= n_nodes <= MAX_NODES`.
2. Node 0 is the only `Input` node, and its `out_dim == n_inputs`.
3. Every input index of node `i` is `< i`. Strict topological order means no
   forward references, no self-references and no cycles, by construction.
4. The **last** node is the graph output: its width is the header `n_outputs`, and
   it is not `Input`.
5. Every node except the last is consumed by at least one later node (no dead
   nodes: a correct exporter never emits one, and dead compute is an attack surface).
6. Op byte, activation byte and dtype byte are known; fields an op does not use are
   zero / empty; `flags` and `reserved` are zero. Unknown bytes refuse at load.
7. Per-op shapes from the table above; Gather indices `< width(input)`; Dense
   section lengths match `in_dim * out_dim` in its dtype, `out_dim` scales (I8
   only), and `out_dim` biases or none.
8. `Σ_dense in_dim * out_dim <= MAX_TOTAL_WEIGHTS`, and the op-weighted element
   work `Σ_nodes width × cost <= MAX_TOTAL_ELEMS` (costs below). Sections may
   legally alias and non-Dense ops carry no payload, so file size does not bound
   compute; these two limits do. (Fix round 1: before it, a 577 KB file of
   Input(65,536) + 1,023 Exp nodes loaded and took ~210 ms per predict.)
9. The liveness-packed scratch arena (below) is `<= MAX_SCRATCH_ELEMS` f32s.

All arithmetic on untrusted sizes is checked (`checked_add` / `checked_mul`);
overflow is `DimensionOverflow`.

### Limits (`zenpredict::limits`)

| const               | value      | why |
|---------------------|------------|-----|
| `MAX_NODES`         | 1024       | the E33 head is 7 nodes, a lowered 4-layer chain is 5 |
| `MAX_NODE_INPUTS`   | 64         | Concat arity |
| `MAX_TOTAL_WEIGHTS` | 16,777,216 | 2^24 multiply-adds per predict; the rev4 production bake (420→128→1) is 53,888, SOTA-944 is 120,960 |
| `MAX_TOTAL_ELEMS`   | 4,194,304  | op-weighted element work per predict (below); the rev4 production bake uses 678 |
| `MAX_SCRATCH_ELEMS` | 4,194,304  | 16 MiB of f32 scratch, checked before allocation |
| `MAX_DIM`, `MAX_BAKE_BYTES` | unchanged | |

`MAX_TOTAL_ELEMS` costs per output element: 1 for `Input` (scaler), `Gather`,
`Add`, `Mul`, `Concat` and the Identity/ReLU/LeakyReLU activations; 32 for `Exp`;
64 for `Softplus` (two transcendentals); a Dense node is charged `1 + activation
cost` per output (its multiply-adds count against `MAX_TOTAL_WEIGHTS`). The
weights follow measured cost: scalar `libm` `expf` is ~30× a vectorized add. Under
both limits a predict is bounded at a few milliseconds whatever the file size.

`MAX_TOTAL_WEIGHTS`, `MAX_TOTAL_ELEMS` and `MAX_SCRATCH_ELEMS` also apply to v3
files after lowering.
No shipped or plausible v3 bake comes near them (a v3 file could previously alias
one weight section across 256 layers); only crafted inputs see a new error.

## Runtime

**Lowering.** A v3 file is parsed exactly as today (same validation, same errors,
same load-time permutations), then lowered to the graph
`Input → Dense(layer 0) → … → Dense(layer N-1)`, each Dense carrying the layer's
activation as its fused activation. v4 and lowered v3 run through one executor.

**Executor.** Nodes run in file order, a fixed deterministic sequence. Each node's
output lives in a slot of one f32 arena; the output node writes straight into the
caller-visible output buffer.

- Input: the v3 scaler loop, unchanged (`scale == 0 → 1`).
- Dense: the v3 code, unchanged — bias copy (or zero-fill when there is no bias),
  `saxpy_matmul_{f32,f16,i8}` under `#[autoversion(v3)]`, the I8 post-scale
  `b[o] + s[o] * acc[o]` (no bias ≡ `b = +0.0`), then the activation. Bit-identical
  to v3 by construction because it is the same code on the same operands in the
  same order.
- Add / Mul: one elementwise loop under `#[autoversion(v3)]`. IEEE add and multiply
  are correctly rounded at any vector width, so the vectorized result equals the
  scalar one bit for bit.
- Gather / Concat: copies.

The brief asked for magetypes on the elementwise ops. They use archmage
`#[autoversion]` instead, the mechanism zenpredict's `simd` feature already uses:
it is codegen-only, keeps `simd` free of a second SIMD dependency, and leaves no
room for per-tier NaN / signed-zero differences in a hand-written select. For two
loops over at most a few hundred floats, against a Dense node that dominates the
forward pass, explicit vectors would buy nothing measurable.

**Scratch.** At load the parser computes each node's last use and packs slots
first-fit into the arena. A node's slot is allocated before its dead inputs are
freed, so an output never aliases an input it reads. The arena length is fixed at
load (`Model` stores it) and bounded by `MAX_SCRATCH_ELEMS` before anything is
allocated. `Predictor::new` allocates it once (infallible, as today);
the new `Predictor::try_new` does the same with `try_reserve_exact` and returns
`PredictError::AllocFailed` instead of aborting. `Model::from_bytes` moves its
owned-buffer copy (up to `MAX_BAKE_BYTES`) and graph tables to fallible
allocation too, since it already returns `Result`. `predict` and the argmin
family stay allocation-free.

For a lowered chain the arena is `n_inputs + Σ hidden widths that are live at
once` — at most `n_inputs + 2·max_hidden`, the same order as v3's two
`scratch_len` buffers.

## Public API (diff)

zenpredict (all additive vs `main`):

```rust
// model.rs
#[non_exhaustive] pub enum Activation { Identity, Relu, LeakyRelu, Exp /*new*/, Softplus /*new*/ }
pub const GRAPH_FORMAT_VERSION: u16 = 4;
pub const EXP_INPUT_CLAMP: f32 = 30.0;
pub const SOFTPLUS_THRESHOLD: f32 = 20.0;

#[derive(Debug, Clone)]
#[non_exhaustive]                       // and every variant is #[non_exhaustive]
pub enum NodeView<'a> {
    Input { width: usize },
    Dense { input: u32, layer: LayerView<'a> },
    Activation { input: u32, activation: Activation, width: usize },
    Gather { input: u32, indices: &'a [u32] },
    Add { a: u32, b: u32, width: usize },
    Mul { a: u32, b: u32, width: usize },
    Concat { inputs: &'a [u32], width: usize },
}
impl NodeView<'_> { pub fn width(&self) -> usize; }

#[repr(C)] #[derive(Clone, Copy, Debug, Pod, Zeroable)] #[non_exhaustive]
pub struct NodeEntry {                  // the wire entry, like LayerEntry
    pub op: u8, pub activation: u8, pub weight_dtype: u8, pub flags: u8,
    pub out_dim: u32,
    pub inputs: Section, pub data0: Section, pub data1: Section, pub data2: Section,
    pub reserved: [u32; 2],
}

impl Model {
    pub fn n_nodes(&self) -> usize;
    pub fn node(&self, idx: usize) -> NodeView<'_>;          // panics if idx >= n_nodes
    pub fn nodes(&self) -> impl ExactSizeIterator<Item = NodeView<'_>>;
    pub fn is_layer_chain(&self) -> bool;                    // true for every v3 file
}
// LayerView and WeightStorage gain Clone (Copy is queued: adding Copy to an
// existing type is a semver break).

// limits.rs
pub const MAX_NODES: usize = 1024;
pub const MAX_NODE_INPUTS: usize = 64;
pub const MAX_TOTAL_WEIGHTS: usize = 1 << 24;
pub const MAX_TOTAL_ELEMS: usize = 1 << 22;
pub const MAX_SCRATCH_ELEMS: usize = 1 << 22;

// wire.rs
pub const NODE_ENTRY_SIZE: usize = 48;
pub const OP_INPUT: u8 = 0; pub const OP_DENSE: u8 = 1; pub const OP_ACTIVATION: u8 = 2;
pub const OP_GATHER: u8 = 3; pub const OP_ADD: u8 = 4; pub const OP_MUL: u8 = 5;
pub const OP_CONCAT: u8 = 6;

// error.rs (PredictError is #[non_exhaustive])
UnknownGraphOp { node: usize, byte: u8 },
GraphInputRef { node: usize, input: usize },            // forward/self reference
GraphShapeMismatch { node: usize, expected: usize, got: usize },
GraphMalformed { node: usize, what: &'static str },
AllocFailed { bytes: usize },

// predictor.rs
impl<'a> Predictor<'a> {
    pub fn try_new(model: &'a Model) -> Result<Self, PredictError>;
}
```

Unchanged: `Model::from_bytes*`, `Predictor` and every predict / argmin entry
(same inputs, outputs and errors), `WeightStorage` variants (consumers match it
exhaustively), `LayerEntry`, `Header`, `FORMAT_VERSION` (still 3: the chain
composer writes it), `Model::scratch_len` (same formula: max of `n_inputs` and
every node width). `UnsupportedVersion`'s message now says the build reads v3
chains and v4 graphs; its fields are unchanged (`expected` stays 3).

`layers()` / `layer()` / `n_layers()` on a v4 file walk its Dense nodes in node
order. For a v4 file a `LayerView` may have **empty `biases`** (bias-free Dense) and
an **`Exp` / `Softplus` activation**, and the Dense nodes need not form a chain from
the features: it is the whole network only when `is_layer_chain()` is true. Tools
that treat `layer(0)` as "the layer that reads the features" must check it, or
refuse v4 as the in-repo tools do. **Open owner decision** (review P2, options:
(a) opt-in v4 loading, (b) loud chain-only accessors, (c) keep and document): this
PR keeps (c)'s behaviour until told.

zenpredict-bake (all additive vs `main`; the crate is unpublished):

```rust
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]                       // and every variant is #[non_exhaustive]
pub enum BakeNode<'a> {
    Input { width: usize },
    Dense { input: u32, out_dim: usize, activation: Activation, dtype: WeightDtype,
            weights: &'a [f32], biases: Option<&'a [f32]> },
    Activation { input: u32, activation: Activation },
    Gather { input: u32, indices: &'a [u32] },
    Add { a: u32, b: u32 },
    Mul { a: u32, b: u32 },
    Concat { inputs: &'a [u32] },
}
impl<'a> BakeNode<'a> {
    pub const fn input(width: usize) -> Self;
    pub const fn dense(input: u32, out_dim: usize, activation: Activation, dtype: WeightDtype,
                       weights: &'a [f32], biases: Option<&'a [f32]>) -> Self;
    pub const fn activation(input: u32, activation: Activation) -> Self;
    pub const fn gather(input: u32, indices: &'a [u32]) -> Self;
    pub const fn add(a: u32, b: u32) -> Self;
    pub const fn mul(a: u32, b: u32) -> Self;
    pub const fn concat(inputs: &'a [u32]) -> Self;
}

#[derive(Clone, Copy)] #[non_exhaustive]
pub struct GraphBakeRequest<'a> {
    pub schema_hash: u64, pub flags: u16,
    pub scaler_mean: &'a [f32], pub scaler_scale: &'a [f32],
    pub nodes: &'a [BakeNode<'a>],
    pub feature_bounds: &'a [FeatureBound], pub metadata: &'a [BakeMetadataEntry<'a>],
    pub output_specs: &'a [OutputSpec], pub discrete_sets: &'a [f32],
    pub sparse_overrides: &'a [SparseOverride], pub compressed: bool,
}
impl<'a> GraphBakeRequest<'a> {
    pub const fn new(schema_hash: u64, scaler_mean: &'a [f32], scaler_scale: &'a [f32],
                     nodes: &'a [BakeNode<'a>]) -> Self;
    // chained setters: flags, feature_bounds, metadata, output_specs, discrete_sets,
    // sparse_overrides, compressed
    pub fn bake(&self) -> Result<Vec<u8>, BakeError>;
}
pub fn bake_graph(req: &GraphBakeRequest<'_>) -> Result<Vec<u8>, BakeError>;

// BakeError (#[non_exhaustive]) gains
GraphInvalid { node: usize, what: &'static str },
ChainActivationUnsupported { layer: usize },   // Exp/Softplus in a v3 chain
GraphRejected(zenpredict::PredictError),       // the parser rejected the composed graph

// JSON: BakeRequestJson gains `graph: Vec<BakeNodeJson>` (serde default empty);
// `layers` becomes serde-default. BakeNodeJson is #[non_exhaustive], tagged by "op",
// and denies unknown keys. Graph nodes take the new #[non_exhaustive]
// GraphActivationJson (identity, relu, leakyrelu, exp, softplus); ActivationJson
// (v3 layers) is unchanged, so the bake crate stays semver-additive.
```

### API shape choices (fix round 1, from review P2)

Each was free to change before the first publish and breaking after:

| question | choice | why |
|---|---|---|
| fields added to a `NodeView` variant later | every variant `#[non_exhaustive]` (and the enum) | matchers must write `..`, so a future LeakyReLU slope or Gather mode is additive |
| `NodeView` traits | `Debug, Clone` now; `Copy` queued for the next minor | `Copy` would cost nothing, but it needs `Copy` on the existing `LayerView` / `WeightStorage`, which `cargo semver-checks` counts as major (`copy_impl_added`); they gain `Clone` now |
| node index type | `u32` everywhere (`input`, `a`, `b`, `inputs`, `indices`) | matches the wire format and lets lists stay zero-copy `&[u32]` |
| width | `NodeView::width()` | one accessor for every op; fields stay where the op needs them |
| wire entries | public `Pod` `NodeEntry`, like `LayerEntry`; the `NODE_OFF_*` constants are gone | one model for both tables; composers build entries with `Zeroable::zeroed()` + field writes, tests use `offset_of!` |
| `BakeNode` growth | `const fn` constructors, every variant `#[non_exhaustive]` | callers outside the crate must use constructors, so new fields are additive |
| graph bake input | dedicated `GraphBakeRequest` (no `layers`, no permutation fields) | the type only admits what a graph bake uses; internally it reuses `bake`'s section validators/writers |
| JSON typos | `deny_unknown_fields` on `BakeNodeJson` | `"bias"` for `"biases"` errors instead of baking a bias-free Dense |

Queued for the next minor (CHANGELOG): merge `GraphActivationJson` into
`ActivationJson`.

Tools that read structure stay chain-only and refuse v4 cleanly:
`append_metadata_utf8` returns `AppendError::UnsupportedVersion`; `zenpredict
repack` exits 1; zenpredict-viz's `parse_bake`, `forward_with_taps`,
`layer_weights` and `znpr2onnx` return/print "ZNPR v4 op-graph bakes are not
supported by zenpredict-viz yet" (`zenpredict_viz::load_chain_model`; review P1:
before the fix they panicked, ran wrong, or exported a wrong ONNX with exit 0).

`bake()` keeps writing v3 chains byte-identically; it now refuses `Exp` /
`Softplus` layers (they would produce a file v3 readers reject). The JSON baker
writes a graph when `graph` is non-empty, else a v3 chain as before; `graph` with
non-empty `layers`, or with `optimize: true`, is refused.

### JSON node spec

```json
{ "schema_hash": 1, "scaler_mean": [...], "scaler_scale": [...],
  "graph": [
    { "op": "input", "width": 4 },
    { "op": "gather", "input": 0, "indices": [0, 1] },
    { "op": "gather", "input": 0, "indices": [2, 3] },
    { "op": "dense", "input": 1, "out_dim": 3, "activation": "relu", "dtype": "f32",
      "weights": [...], "biases": null },
    { "op": "dense", "input": 2, "out_dim": 3, "activation": "exp", "dtype": "f32",
      "weights": [...] },
    { "op": "mul", "a": 3, "b": 4 },
    { "op": "dense", "input": 5, "out_dim": 1, "activation": "identity", "dtype": "f32",
      "weights": [...] }
  ] }
```

That is the E33 arm-B gated head. Its output is exactly `+0.0` whenever the `d`
slice is zero and every `r` is finite: the bias-free ReLU branch is `0`,
`0 · exp(clamped) = 0`, and the bias-free final Dense skips zero inputs. An infinite
`r` can make `u·r` NaN (∞ · 0 weight, or +∞ − ∞), and NaN propagates. The `d`
features must be zero *after* any feature transform and the scaler, so a gated
head is baked with `mean = 0` on the `d` columns (or the trainer folds the mean
into the branch) and transforms that map 0 to 0 there.

## Migration

- v3 files: load unchanged, run through the graph executor, bit-identical outputs
  (gated by the cross-version parity harness, below).
- v2 files: still rejected with `UnsupportedVersion`, as v3 readers have been since
  2026-05 (`docs/ZNPR_V3.md`, owner direction "everyone uses v3"). The ZPGRAPH brief
  asked for v2 to load as a chain; that would turn today's error into a success and
  contradict the v3 contract, so it is left for the owner to decide.
- v4 files: rejected by every earlier zenpredict with `UnsupportedVersion { 4, 3 }`.
  Consumers must update zenpredict before loading a graph bake. Chain bakes stay v3,
  so nothing changes for pickers or the shipped zensim bakes.

## Verification

- `zenpredict/tools/graph-parity/`: a standalone crate that links the pre-graph
  zenpredict (git `main` before this change) and this one side by side, and checks
  bit-identical `predict` / `predict_transformed` output on every ZNPR file it is
  given (repo fixtures, zensim's shipped bakes, the rev4 production bake) over
  random and adversarial feature vectors, plus identical errors for files that fail
  to load. The same crate runs the paired zenbench old-vs-new forward benchmark.
- Unit and integration tests: graph validation (every rule above), lowering, the
  gated-head zero test, Exp/Softplus math, bake → load round trips.
- Fuzz: `graph_from_bytes` (raw bytes, seeded with real v4 bakes) and
  `graph_structured` (an `arbitrary`-built node table). `graph_structured` is
  differential: every graph that loads must match the per-node reference
  evaluator in `fuzz/fuzz_targets/graph_core.rs` (bits, or NaN for NaN), the same
  evaluator `zenpredict-bake/tests/graph.rs` uses.
- Exp / Softplus golden bits: `inference::activation_tests::exp_and_softplus_golden_bits`
  asserts fixed hashes of the output bits over a 5,121-point grid plus edges, on
  every CI target.
