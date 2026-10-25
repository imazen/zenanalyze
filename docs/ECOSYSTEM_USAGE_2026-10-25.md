# Ecosystem use and approaches to reconsider

Audited 2026-10-25. Keep the extractor, transport contract, MLP runtime and
baker. The weakest part is the qualification of old learned selectors and the
amount of speculative controller API around them. Publication status alone
does not distinguish those two things.

This is a source-use and focused-test audit. It does not establish deployment
frequency, a rate-distortion improvement, or a replacement model's qualification.
Recommendations below are proposals; no codec, model or test expectation changed.

## Scope and evidence

The census discovers Git repositories on dev, reads local and origin branch
tips plus jj working-copy snapshots, parses Cargo manifests, resolves renamed
dependency module names, and records source references. A separate manifest
graph includes indirect codec/metric dependencies. It scans immutable commits,
leaving working files alone. Relevant origin refs were refreshed through jj.

The broad inventory found 168 object stores. Detailed source/symbol passes
covered the dependent codec, metric, pipeline and tooling repositories. Dev's
busy zenjpeg checkout was supplemented from the clean i265 repository's 95
remote branch tips. Live uncommitted edits remain outside the audit. Older
zenpng/oxcms objects had read errors; these are recorded rather than treated as
negative evidence. Static indirect edges include optional and tool dependencies;
they are not a claim that every edge is enabled in a product's default build.

The [evidence pointer](../benchmarks/dependency_usage_2026-10-25.pointer.md)
records the private census and its hashes. The tool is
[`audit_dependency_refs.py`](../tools/audit_dependency_refs.py).

## What is substantially used

| Component | Observed consumers | Assessment |
|---|---|---|
| zenanalyze extraction and narrow feature requests | zenjpeg adaptive encoding; default-on jxl-encoder learned admission; fleet source-feature export; opt-in WebP classifier, AVIF tuning and SR chooser | Core extraction is substantially integrated. Golden/numeric tests qualify extraction; they do not requalify fitted models after feature drift. |
| zenanalyze-api Request/Offer, qualified identities and Names | WebP classifier, AVIF/JPEG offer reuse, SR chooser and zencodecs picker interfaces | Keep the small transport contract. Version-agnostic selection suits threshold heuristics; compiled models need their training identities. |
| zenpredict Model/Predictor and v3 parsing | zensim metric runtime, training/validation tools, AVIF and JPEG pickers; optional codec research paths | Keep the runtime. Six inspected artifacts all use v3; f16, f32, i8 and payload compression occur in those artifacts. |
| zenpredict-bake | zensim training/validation, zenpicker-train, zentrain's bake CLI invocation, tests and inspection | Substantially used tooling. Keep it separate from inference dependencies. |
| zenpicker family types, masks and heuristic fallback | coefficient family types; optional zencodecs routing | Keep these uses distinct from the learned routers. The current learned routes refuse current analyzer identities. |

Source examples: [jxl admission's four-feature request](https://github.com/imazen/jxl-encoder/blob/ce4ec7a8e4bf4a74a0b55a1851bc5b2b84167640/jxl-encoder/src/vardct/learned_admission.rs),
[AVIF's model-specific request](https://github.com/imazen/zenavif/blob/a7638f6e0655914fb15fe4633ac26425533756c8/src/auto_tune.rs),
[fleet source export](https://github.com/imazen/zenmetrics/blob/bd08285eba2685867c1368955ee113e259b6bfff/crates/zenfleet-vastai/src/worker/source_features.rs),
[zensim runtime bridge](https://github.com/imazen/zensim/blob/51467404c65544957668eeeb9b4fdc9216a4e95b/zensim/src/mlp/mod.rs).

Indirect paths also reach imageflow and its server/bindings, zensquoosh,
codec-eval, ultrahdr, zentiff, zensysbench, squintly, glassa and codec/metric
test tools. Their codec or metric dependency does not make them direct users
of the picker APIs. Zenjxl's direct analyzer use is in its feature-extraction
tool; its encoder dependency also brings jxl-encoder's admission path.

The indirect manifest map also found these concrete roles. Feature activation
and dependency version resolution still need a consumer build before a cut:

| Role | Repositories and observed paths |
|---|---|
| Applications and codec wrappers | imageflow/imageflow-server-rs/imageflow-node through codec features; hdr-research's editor through JPEG; zensquoosh through zencodecs; ultrahdr-rs through JPEG; zentiff through optional JPEG; jxl-encoder-gpu and jxl-gpu through optional CPU encoding |
| Evaluation, tuning and research | codec-eval codec-compare/codec-iter; glassa through optional coefficient; squintly through codecs; zensysbench through optional codecs; dvifmish's optional JPEG/JXL paths; aom-rs benchmark tools through optional zensim; butteraugli's margarine experiment through zensim |
| Examples, compatibility and regression tools | all-the-images test harness; imazen-26 tools; archmage downstream compatibility; webpx's zencodec-swap demo; rav1d-safe's Rust-codec workflow; zencodec's JPEG policy/streaming regression tools; zentone's dev shootout |

Private tool consumers remain in the private evidence. These indirect edges
were not counted as new direct users of rescue, confidence or graph APIs.

## Model qualification findings

1. **The meta-routers are effectively disabled for current analyzer offers.**
   The existing live test produced **0 model routes and 75 explicit heuristic
   fallbacks**. It identified covariance Cb/Cr and chroma subsampling loss as
   the mismatched columns. The fallback behavior is tested; learned routing
   quality on current features is not demonstrated by that passing test.
2. **JPEG's bake declares feature-definition version 1; the extractor emits 2.**
   Its 108 source inputs include both covariance columns, and its feature-bounds
   array is empty. The reuse code qualifies an offer against the current
   extractor's hashes, not hashes captured from training. Agreement between an
   offer and a fresh extraction therefore does not establish agreement with
   the model's training definitions. This is a qualification gap, not a
   measured rate-distortion regression in this audit.
3. **AVIF has 6 unavailable inputs out of 43 baked source columns.**
   `text_likelihood`, `screen_content_likelihood`, `natural_likelihood`,
   `line_art_score`, `log_min_dim`, and `log_max_dim` are absent from the
   current supported set. The own-pass and reuse paths fill unavailable
   columns with zero. The bake has 96 input bounds, but `auto_tune` does not
   call the bounds checker. It has no covariance columns: the concrete
   missing inputs and any changed inputs it actually uses matter more than
   blanket conclusions from the global definition counter.
4. **The SR chooser is directly exposed to covariance drift.** Its trained
   21-feature logistic rule contains both covariances with nonzero weights.
   It requests All, looks up bare names and records no training feature hashes.
   Its source already documents that limitation. Retaining the same column
   names cannot establish that the old coefficients remain qualified.

Sources: [JPEG picker](https://github.com/imazen/zenjpeg/blob/de26db2f20c2290b7f4f93ea2377a14bcbec5c6d/zenjpeg/src/encode/picker.rs),
[AVIF own-pass and reuse](https://github.com/imazen/zenavif/blob/a7638f6e0655914fb15fe4633ac26425533756c8/src/auto_tune.rs),
[SR chooser](https://github.com/imazen/zensr/blob/1cbc449216d9da2a0d7e4ee5ac206277ff114d08/crates/zensr-zenjpeg/src/chooser.rs).
The artifact inspection used the existing zenpredict-inspect binary; the
feature-name comparison used the current list_features output and inventory
tool. No unavailable input was silently treated as a successful qualification.

The analyzer's image-content features and zensim's perceptual feature banks
are different inputs. Sharing zenpredict does not make their definitions or
training data interchangeable. The inspected B rev5 artifact uses the latter.

## Branches that affect the decision

- **feat/znpr-v4-graph** (`84800dd1`): an implemented, unmerged DAG runtime
  for gated/multi-branch heads, with tests and fuzz targets. Its motivating
  head is documented in [the v4 design](https://github.com/imazen/zenanalyze/blob/84800dd1feb2/zenpredict/docs/ZNPR_V4_GRAPH.md).
  It is not a prerequisite for the inspected v3 artifacts. It also needs a
  wire compatibility repair before landing: activation 3 means Exp in its
  v4 graph, and its v3 parser rejects 3; main now uses v3 activation 3 for
  SmoothHinge. Version-specific mapping and current-v3 regression coverage
  are required. Defer it unless a qualified head needs a DAG.
- **feat/zenpicker-rescue-on-top-k** and **feat/zenpicker-safety-gates**:
  diverged prototypes. No out-of-workspace calls to the current generic
  rescue, knob-veto or unachievable-zone helpers were found in the symbol
  census. Preserve the experiments; do not make their generalization a
  release prerequisite.
- **zencodec feat/zencodec-helpers-loop-tools-experimental** (`7fae330f`)
  calls masked top-K selection. Its sibling picker-topk branch (`7c6a6c37`)
  has a separate selection prototype. Both were one commit ahead and 262
  behind main at comparison. These are prototype consumers, not default
  product use. Current zencodecs does call argmin_masked_with_scorer.
- **steer-smooth-hinge**: its functionality is present on main and is used
  by zensim training/gradient code and registered fit checks. SmoothHinge is
  not an unused extension to remove alongside older picker experiments.
- **zenjpeg archive/diffmap-rd-zq-experiment-2026-07-18**: current main has
  the later complete candidate binding. Use that path's per-call BakeScorer
  and current reconstruction maps rather than reviving the archived global
  profile/cache/controller assumptions. See [candidate binding](https://github.com/imazen/zenjpeg/blob/de26db2f20c2290b7f4f93ea2377a14bcbec5c6d/docs/zensim-candidate-binding-2026-09-08.md).

## Approaches to ablate or narrow

| Approach | Proposed disposition | Gate before changing behavior |
|---|---|---|
| Old one-shot full-config byte predictor as final authority | Compare retiring this role; retain a predictor as an optional seed/shortlist | Matched-feature baseline, heuristic control and verified native-loop control at equal observed encode budgets |
| Learned family routing with stale bakes | Keep reported heuristic routing as the baseline; retrain or explicitly gate the learned layer | New training identities, live engagement, and held-out routing RD; model parsing alone is insufficient |
| Broad per-cell outputs and every available image feature | Ablate output knobs first, then correlated input groups; request only the surviving inputs | Content/quality/size/mode coverage, grouped importance and retraining, followed by actual encoded RD |
| Generic rescue/confidence/safety scaffolding | Candidate for an experimental surface or pre-publication removal of unused wrappers | Consumer/branch builds, frozen artifact replay and explicit API review; retain wire metadata compatibility and necessary OOD checks |
| Full-budget extraction as a remedy for picker errors | Keep out of the default path; deprioritize further expansion | Fix identity/size-domain problems first; more samples do not repair mismatched coefficients |
| Arbitrary v4 graph generalization | Defer until a qualifying model requires it | Current-v3 compatibility, actual head benefit, and runtime/derivative qualification |
| coefficient's unconditional analyzer dependency | Candidate to move to tools/dev or a specific feature | Main's normal src has no analyzer references in the census; build its examples, scripts and product targets before cutting |
| Adaptive analysis as an unconditional JPEG dependency | Evaluate a separate adaptive-analysis boundary if a minimal basic-encoding dependency graph is wanted | Gating an existing public method is an API decision requiring review; it does not make the extractor obsolete |
| Ranking-only selection of an optimizer metric | Retire as a sufficient qualification rule | Also assess distortion response, mixture/knob inversions, targeting feasibility and actual codec outcomes |

Do not ablate scalar/FMA correctness, versioned feature identities, parser
limits, strided input support or explicit fallback reporting. Keep f16/f32,
v3 payload compression and the active SmoothHinge path. The six-artifact
sample does not justify removing other wire capabilities or transforms.
Performance gains from the proposed API/dependency cuts were not measured.

Use the existing [picker ablation owner/design](https://github.com/imazen/zenmetrics/blob/bd08285eba2685867c1368955ee113e259b6bfff/docs/ML_FRAMEWORK_AND_PICKER_ABLATION_2026-06-09.md)
and [zensim-target controller](https://github.com/imazen/zensim/tree/70812747f52a/zensim-target)
for comparisons. The latter remains an experiment/controller owner with
documented limitations, not automatic proof of a production replacement.
New metric candidates are not promoted by this audit.

The next experiment should compare heuristic-only, a matched legacy
model/extractor, a smaller current-definition seed model, and the existing
verified controller. Freeze selection on TRAIN/development data; reserve the
final assessment. Include small through 4K images, aggressive qualities as
densely as high qualities, content classes and every relevant codec mode.
Judge target error, bytes, independent quality and observed work at matched
budgets. No new RD or ablation winner was measured here.
