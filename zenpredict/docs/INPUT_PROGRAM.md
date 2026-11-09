# Required input composition

This proposal extends ZNPR with products, zero-anchored damage shaping and
reference gates. It builds on the analytic activations and stable max-pair
proposals. No frozen serving model or existing v3 score changes.

## Serving contract

`Predictor::predict_from_source` runs:

1. Read source columns in the producer-declared order.
2. Execute the ordered input program and gather its output nodes.
3. Apply existing nonlinear feature transforms (including expanders).
4. Apply the embedded fixed scaler.
5. Run the existing dense/activation engine.

`Model::n_source_inputs()` describes the first step. `n_inputs()` still describes
the scaler and first dense layer. `predict` and selection wrappers retain their
scaler-only contract for caller-prepared inputs; `predict_transformed` starts at
step 3. They do not implicitly compose products. These domains can have equal
widths: length checking cannot prove that a caller chose the correct domain.
For source inputs use `predict_from_source`, then a free selection function.
`predict_with_specs_from_source` also applies output specs and overrides.

Source IDs, extraction revision, physical meaning, train split and scaler
provenance remain producer responsibilities. Pin `schema_hash` on load. A
metric exporter must still verify that its product factors have the intended
semantic role, scale and channel. The generic runtime cannot infer those facts.

## Program

`zenpredict::input_program::{InputProgram, InputOp, DamageShape, ReferenceGate}`
are usable without std or serde. A node is a Source, Product, Shape or Gate.
References address earlier nodes; output references select compact model inputs.
This expresses both `shape(damage * reference)` and
`shape(damage) * gate(reference)` without combinatorial variants or hidden order.
The graph is a straight-line program; cycles and forward references are errors.
Common subexpressions can be reused. No allocation occurs during evaluation.

Damage shaping clamps negative inputs to zero and integrates three nonnegative
slopes over two positive ordered knots. Identity slopes return the clamped input
exactly, including negative zero. The arithmetic order is retained from the
research wire implementation; no FMA or reassociation is introduced. Reference
gates interpolate values at 0, 0.5 and 1 and clamp outside that interval. Gates
are nonnegative, but need not be monotone. Product monotonicity needs appropriate
factor domains; the program alone does not certify a whole network as monotone.
NaN/infinity follow the documented floating-point operations. No finite-output
guarantee is implied.

## Producer paths

Rust producers call `zenpredict_bake::input_program::bake_with_input_program`.
`attach_input_program` adds or replaces composition on an existing bake without
reordering hidden units or requantizing weights; the runtime parser validates
the resulting model. This is the preferred migration path for frozen research
bakes when exact network arithmetic matters.

JSON producers add an optional top-level `input_program`:

```json
{"source_count":2,"nodes":[
  {"op":"source","index":0},
  {"op":"source","index":1},
  {"op":"product","left":0,"right":1},
  {"op":"shape","input":2,"knots":[0.25,1.0],"slopes":[2.0,0.5,3.0]}
],"outputs":[3]}
```

This field accepts the object above; it is not a complete bake request. Shape
and Gate parameter fields reject unknown spellings. `zenpredict bake` uses the
same typed decoder as the Rust JSON API. `zenpredict inspect` exposes source
count and typed program nodes as well as the original metadata bytes.

## Wire format and compatibility

Legacy bakes still emit version 3 with unchanged arithmetic. Bakes containing
required composition emit **version 4**, with the same 128-byte physical header,
section tables, layer records and compression envelope. Version 3-only readers
reject version 4 before inference. A reserved header flag or opaque v3 metadata
alone would be insufficient because older readers ignore those extensions.
Version 4 requires exactly one bytes entry named `zenpredict.input_program`;
the new reader rejects this key in a version 3 bake, duplicates, wrong types and
output arity mismatches. The default `zenpredict::FORMAT_VERSION` remains 3 for
ordinary bakes; `input_program::FORMAT_VERSION` is 4.

The entry value is little-endian:

- 8 bytes: `ZINP`, u16 schema version 1, u16 zero flags.
- Three u32: source count, node count, output count.
- Each node: u8 opcode, three zero bytes, then u32 node/source indices and f32
  parameters. Source=0 (index); Product=1 (left,right); Shape=2 (input,k0,k1,s0,s1,s2);
  Gate=3 (input,g0,gmid,g1).
- Output count u32 node references; no trailing bytes.

Counts are bounded by MAX_DIM; byte availability is checked before allocations.
Reserved bytes, unknown codes/schema, nonfinite/negative shape parameters,
invalid knots, references and trailing bytes fail closed. Metadata is parsed at
load, with per-predictor scratch and no per-call parsing. Whole-bake compression
and caller-order feature/output permutations retain their existing meaning.

## Review findings and qualification scope

The input-transform and output-spec paths had duplicated execution code. Both
now use shared forwarding and output finalization. The added source path uses
those same helpers rather than a metric-private scorer. Existing semantics for
caller-prepared prediction remain explicit and tested.

Hidden-unit pair declarations refer to semantic units. The baker now remaps
those indices together with first-layer hidden-unit permutations. Previously
an automatic reorder could make a valid declared pair unloadable. Explicit
hidden-unit permutations are validated before indexing; empty optimized requests
return EmptyLayers instead of indexing an empty slice.

Automatic HU reordering is mathematically equivalent but can alter floating-point
accumulation order in later layers. Existing composer defaults remain unchanged;
attach composition to the original bake when bitwise research parity is required.
No optimizer speed or model-quality improvement is claimed by this change.

Remaining boundaries: arbitrary nonlinear programs are not symbolic monotonicity
proofs; generic product operators do not validate metric feature semantics;
existing nonfinite input behavior is retained; source/prepared input domains are
explicit API choices. Full dataset evaluation and model promotion require their
own frozen exposure plan. This format implementation does not open holdouts.

The current visualizer has a separate diagnostic trace engine. It does not
implement input programs, Softplus/SiLU or stable max-pair tracing; these models
now return an explicit trace error instead of displaying identity activations
or omitting composition. Native runtime prediction supports them. The ONNX
exporter also rejects required input programs rather than exporting a partial
model as if composition were included. Trace/export support is a separate task.

Independent research-wire qualification loaded three saved bakes (frozen product,
joint product and separate joint shaping) and compared 64 deterministic source
vectors per bake. All 192 compact vectors and prediction vectors matched bit for
bit. The oracle was the original research `derived_inputs/shaping.rs`, compiled
alongside the consumer API. No dataset labels were loaded. Reproduce using
`just predict-research-shaping ORACLE OUTPUT "BAKES..."`; the tool preserves exact
bakes, oracle source, vectors, generated harness and SHA-256 receipts outside git.
These are migration parity cases, not new model-quality evaluations.

The review also repaired Drop-only preprocessing without parameter metadata and
sizes expander parameter scratch by raw transform count. A program may output
more raw compact fields than the post-Drop network width; this must not force an
allocation on the first prediction or falsely require Sinusoidal frequencies.

Baker and optimizer dimension preflight is shared: empty/zero/excessive dimensions
and overflowing weight counts return errors before indexing or allocation.
