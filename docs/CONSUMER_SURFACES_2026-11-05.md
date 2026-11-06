# Correctness follow-up and consumer surfaces

Source inspection on 2026-11-05: zenanalyze `38dc80f7` plus the directed-search
repair in this session; clean local zensim `8a95a64`, zensimx `506238280`, and
zensr `1cbc449`. These are inspected checkout revisions, not a claim that all
consumer changes have been published. Consumer checkouts were read without
synchronization or edits. This updates the
[October API audit](API_NECESSITY_2026-10-28.md).

Follow-up on 2026-11-06 UTC: PR 92 merged as `c3fb486` and its controller
changes shipped in zenpredict 0.2.0. `resolve_pre_argmin` is experimental,
takes an independent hard mask and refuses a forbidden or out-of-range zone
fallback. The original proposal discussion below describes the design rationale;
this signature is now implemented. Consumer checkout observations remain the
dated source inspection above, not a new consumer qualification.

## Correctness disposition

| Finding | Disposition | Reason and remaining work |
|---|---|---|
| no_std Exp/sigmoid and inaccurate Round | Already fixed on main in `7064f1df` | libm implements the no_std math; rounding preserves large integral values, signed zero and infinities. Existing nonzero/boundary controls remain enabled. |
| NaN wins trial selection | Fixed in `b15e914a` | Every winner branch excludes NaN measurements. Search excludes NaN predictions and measured evidence, keeps failed candidates tried, and refuses a NaN quality target. Entirely invalid trial sets return `None`; infinities retain numeric ordering in winner selection. New controls failed before repair. |
| Zone fallback can bypass a caller prohibition | Fixed in merged PR 92, zenpredict 0.2.0 | The experimental helper takes an independent hard mask and accepts a zone fallback only if its index is in range and permitted. It may bypass predicted quality reachability, but cannot widen hard constraints. The caller still owns scalar constraints and later rescue. |
| Raw prediction bypasses baked nonlinear transforms | Intentional primitive plus misleading usage guidance | `predict` and Predictor argmin wrappers apply the scaler only. Preserve the existing raw-path regression. Documentation now directs raw extracted features through `predict_transformed` plus a free selector. Consumers must choose the correct input domain. |
| Feature offer configuration/framing | Compatibility precondition, not enforced by the frozen request | `Select::Features` checks qualified identities only. Model training configuration and crop/framing must be checked separately using existing provenance, or established when grouping extraction passes. Do not silently change frozen request semantics. |
| Packed-only offer producers | API design limitation | Core extraction already takes a strided `PixelSlice`. A fallible producer accepting that slice and an `AnalysisQuery`, deriving framing from the source, would preserve the existing request resolver and result conversion. Review an additive producer entry before implementation; do not replace existing signatures unilaterally. |
| Zero trial budget / empty candidate set resolves to one | Existing specified policy | It is an allowance around a mandatory encode. Keep the existing tests. A truly hard total-work ceiling would need a distinct contract. |

The zone helper now distinguishes "caller forbids this setting" from "this
setting is predicted to miss quality" through separate masks. It validates a
fallback against the hard mask; otherwise it returns to argmin with hard
constraints applied. An empty resulting mask means no candidate. Codecs still
validate scalar limits and preserve hard constraints through any later rescue.

There is also a consumer-specific model qualification gap in zensr, described
below. A feature flag cannot correct mismatched training definitions.

## What the consumers actually need

| Consumer and path | Analyzer / contract | Predictor runtime | Additional tooling |
|---|---|---|---|
| zensim serving | No direct zenanalyze/API dependency | `Model::from_bytes`, metadata and input-width access, transforms/params/arity, first-layer weight views, `WeightStorage`, `f16_bits_to_f32`, `Predictor::{new,predict,predict_transformed}` | zenpredict-bake is a dev-dependency for synthetic tests. |
| zensimx serving/custom bakes | Its metric features are distinct from analyzer content features | Same serving/planning primitives; custom derivative/scoring paths additionally traverse layers, read scaler arrays and match activation/transform variants | Full model inspection is also consumed by the separate baker. |
| zensimx training/validation | Some controller/codec experiments pull analyzer dependencies; not a requirement of the base predictor | Layer/scaler/transform inspection for pruning, refitting, attribution and exact derivatives; ordinary prediction for replay | `zenpredict-bake::{BakeLayer,BakeRequest,bake,bake_from_json_str,append_metadata_utf8}`, dtype/activation/metadata enums and bake tools. |
| zensr chooser | `offer_for_request`, `Request::new(Select::All)`, borrowed/owned offers, feature lookup and native-to-f32 projection | No direct zenpredict dependency or call in the SR chooser | Bench tools also use the concrete analyzer's typed query/result surface. A transitive codec picker dependency is not direct chooser use. |
| zenpicker | Borrowed Offer/Request/Select, qualified names, provenance-bearing feature results; no normal dependency on the analyzer implementation | Model loading/metadata/feature columns, Predictor and argmin, family masks, `EncodeMode`; `MultiCellPicker` explicitly selects transformed prediction when needed | zenpredict-bake is a dev-dependency. `advanced` is not needed by zenpicker's normal dependency. |

Production zensim genuinely needs **weight inspection**. In
`zensim/src/fold_engine.rs:225`, `caller_col_spans` calls transform
`output_arity`; `caller_line_reads` at line 266 reads `model.layer(0)` and
matches f32/f16/i8 storage. `feature_plan::bake_read_slots` uses those results
to select required feature work. Treating this as training-only would break
serving feature planning. Its production forward path explicitly chooses
`predict_transformed` when required (`metric.rs:2583`).

The current production model was read through the canonical
`zenpredict-inspect`, without writing or rebaking the artifact:

- File: `zensim/weights/rev7_mix_h128_2026-11-03.bin`.
- SHA-256: `d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`.
- File size: 24,504 bytes; ZNPR v3, flags 3 (LZ4 payload envelope).
- Shape: 984 → 128 → 1; first layer f16/ReLU, final layer f32/Identity.
- No `zentrain.feature_transforms` or transform-parameter metadata.
- Derived input/layout metadata is interpreted by zensim, not by a new
  zenpredict inference backend.

This model alone does not require SmoothHinge or Sinusoidal; other validated
models and the public loader may still do so. It also demonstrates that f16,
f32 and compressed-v3 support belong to the serving dependency.

zensimx's extra derivative tape lives in
`zensim/src/metric/bake_gradient.rs`, using `Model::layers`, scaler arrays,
`Activation`, `FeatureTransform` and `WeightStorage`. The tape and optimizer
are not implemented in zenpredict. The training binaries live under
`zensim-validate`; `zensim-train-core` consumes the separate baker. None of
those inspected normal manifests opts into zenpredict's `advanced` feature.

## Feature-flag recommendation

**Keep the default model-reading/planning surface; do not introduce a blanket
`zensimx` or `research` flag around it.** Both metric repositories need much of
it for inference, and the extra research algorithms already live outside
zenpredict. The baker is already a separate dependency, so serving does not
pull its JSON/composer machinery.

A narrower future `model-inspect` flag could expose the convenience
`Model::layers`/`LayerIter`, scaler accessors and transform token/application
helpers for the baker and research tools. It would have to preserve the
single-layer/arity views production zensim needs. Several of the same methods
are already required internally by inference and parsing, so such a flag
primarily changes public visibility; no binary-size or compile-time saving was
measured. The remaining generic views can express much of the same access.
That is insufficient benefit to add a flag and consumer migration in this fix.

The existing `advanced` gate covers typed output postprocessing, confidence,
closure-scorer, rescue and compact safety APIs. Neither metric's base serving
path needs those APIs, and neither zensr's chooser nor zenpicker currently
needs them. zencodecs' closure-scorer caller, established in the earlier audit,
still needs its capability. Do not relabel the whole advanced gate as
zensimx-specific or remove it based on this four-consumer sample.

Keep wire parsing and valid-model numerical behavior independent of API-only
gating. If a future computational capability becomes optional, loading a model
that needs it must refuse explicitly; silently substituting identity/default
math would repeat the no_std correctness defect.

## Consumer-specific risks to carry into release review

- **zensr:** its chooser is a fitted 21-feature logistic rule over a center
  crop capped at 512×512. It requests All, reads bare names and replaces absent
  features with training medians. Its own source records missing training
  feature hashes. Both changed chroma-luma covariances have nonzero coefficients.
  Do not manufacture identities from the current extractor or switch to Names
  as a purported correctness repair. Recover the training extractor/provenance
  and revalidate or refit the rule. An offered full-image analysis is not the
  calibrated crop. This is a verified qualification gap; no routing degradation
  was measured in this follow-up.
- **zenpicker:** the old shipped routers remain identity-mismatched; explicit
  heuristic fallback is the established behavior. `MultiCellPicker` handles
  transforms itself. Legacy `MetaPicker::pick`/router scoring call the raw
  forward path; a new transform-bearing router needs correctly prepared input
  or a reviewed transformed composition. Do not retrain or alter current
  fallback policy as part of a generic runtime fix.
- **Dependency validation:** zensim and zensimx still pin predictor revision
  `f0903489`; zenpicker uses the in-workspace path. Zensr's workspace patches
  the analyzer contract from Git and resolves it through its lockfile. Source
  inspection is not a complete consumer build or a model requalification
  against current main. No sibling dependency, model, threshold or public
  signature changed here.

## Verification

Directed-search invalid-input tests failed before repair, then passed alongside
all pre-existing directed tests. Runtime no-default all-targets, all-feature
all-targets, all-feature doctests and strict Clippy pass using an isolated
runtime/baker/API workspace. Existing doctest ignores are unchanged. The
canonical inspector successfully read the production artifact. New documentation
examples are compiled by the runtime doctest gate.

Full logs and artifact inspection are retained at
`/mnt/v/output/zenanalyze/correctness-consumers-2026-11-05/` on dev; the
`SHA256SUMS` file binds the saved evidence. No consumer quality, performance,
memory saving or new feature-gate compatibility result is claimed.
