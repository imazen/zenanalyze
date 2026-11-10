# Stable research max-pair evaluation

Caller: zensimx's registered activation/shaping completion study. Its dense
ReLU max-pair representation can reverse damage order after float32 rounding.
One archived checkpoint increases its score by 0.0000152587890625 on a
registered damage probe. Scalar fused arithmetic reproduces that reversal.
This opt-in operator evaluates the same real-arithmetic pair function directly.
Existing unmarked models retain their original evaluation.

The first layer uses `Activation::ResearchReluMaxPairs` (wire code 6; JSON
`research_relu_max_pairs`). Older runtimes reject that code. Required UTF-8
metadata `zenpredict.research_max_pairs` has this form:

```
relu-max-pairs-v1
0 1
4 7
```

Indices refer to the stored first-layer units after any hidden-unit permutation.
The declaration must be attached with correct indices; changing a permutation
requires remapping them. The parser rejects missing metadata, empty/overlapping
pairs, out-of-range indices, use after the first layer, incompatible layer types,
nonfinite weights, untied following-layer rows, and invalid detector constraints.
`PredictError::InvalidResearchMaxPair { reason }` reports invalid declarations
or input domains. These are additive Rust API and wire changes for review.

For each pair `(u,v)`, the existing first layer represents
`ReLU(a·x) + ReLU(c·x−t)`, with exactly tied outgoing weights. Require `a≥0`,
`a+c≥0`, zero bias on `u`, nonpositive bias `−t` on `v`, and nonnegative scaled
inputs. The scaler must have zero means and finite positive scales. Both the
first and following layer store f32 weights. Then the pair equals
`max(a·x, (a+c)·x−t)`. The runtime reconstructs the second detector coefficients
and accumulates both detectors in f64, casts their maximum to f32 in `u`, and
sets `v` to zero. Other units and subsequent dense layers keep their existing
arithmetic. The coefficients, architecture's mathematical function, and training
objective do not change; rounded scores can change deliberately.

This establishes a stable evaluation of the declared pairs, not a claim that an
arbitrary whole network is monotone. Whole-model constraints, native prediction
parity/quality, identity and order probes, numerical boundaries and inference
cost must still be assessed. Research exports must retain original bakes and
record new model hashes. No serving model is promoted by this change.

The compact regression exercises a real float32 order reversal, adjacent-float
monotonicity around the witness, identity, invalid metadata and negative inputs.
The broader qualification command is `just analytic-activation-check`.
