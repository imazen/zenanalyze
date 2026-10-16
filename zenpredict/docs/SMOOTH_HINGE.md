# Fixed smooth hinge activation

The steering experiment in zensim needs a convex, nonnegative C1 hidden activation
that is exactly zero at zero. ZNPR v3 layer activation byte 3 declares
`Activation::SmoothHinge`, with tau fixed at 1/8 (not a learned or implicit field).
JSON baking and inspection use `smooth_hinge`. Codes 0/1/2 and their arithmetic
are unchanged. Old readers refuse code 3 as UnknownActivation; they cannot
silently evaluate the model as ReLU. No model format version/header changes.

For z <= 0 the value is zero; on (0,1/8) it is 4*z*z; above that it is z-1/16.
The mathematical derivative is 0, 8*z, 1 on those intervals; it agrees at both
endpoints. This keeps h>=0 and h(0)=0. Nonpositive output weights preserve raw
concavity and a zero-bias identity pin in zensim's N head. It does not establish
finite-edit ranking or perceptual quality.

Public API delta: one variant of the existing non-exhaustive Activation enum,
and one ActivationJson variant; no new functions or signatures. Canonical baker,
parser, execution, inspector and visualization recognize the fixed wire code.
ONNX export remains an explicit unsupported-activation error for this code.
