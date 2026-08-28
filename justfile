# zenanalyze dev commands

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# The light zentrain tool tests — same file list as the CI `zentrain-pytests`
# job (numpy / sklearn / pyarrow only, no torch). Run from zentrain/tools so
# the sibling modules import by name.
zentrain-pytests:
    cd zentrain/tools && uv run --no-project --with pytest --with numpy --with pyarrow --with scikit-learn --with joblib \
      python -m pytest -q test_provenance.py test_predict_lib.py \
      test_picker_lib_strict.py test_metapicker_lib.py \
      test_train_hybrid_knob_vetoes.py test_cargo_invocations.py \
      test_train_hybrid_backend.py test_feature_inventory.py \
      test_canonical_tools.py \
      --deselect test_predict_lib.py::test_student_permutation_relu \
      --deselect test_predict_lib.py::test_student_permutation_leakyrelu

# Regenerate docs/feature-consumption.md (zenanalyze#41): which analyzer
# features each downstream bake consumes, read from the bake artifacts. The
# universe is this build's FeatureSet::SUPPORTED (experimental default-on,
# plus hdr). Sibling codec checkouts default to the ~/work/zen layout;
# override with `just feature-inventory zen=/path/to/zen`.
# Per-feature cost grid (solo / leave-one-out ns per class × side × crop ×
# feature) on real codec-corpus crops at 64²..4096² — the input for
# `feature-inventory`'s cost-vs-use section. ~5 min; writes
# benchmarks/per_feature_cost_grid_<date>.tsv (set PFC_OUT to override).
per-feature-cost-grid corpus="../codec-corpus":
    ZENANALYZE_CORPUS_DIR={{corpus}} cargo run --release --features hdr --example per_feature_cost_grid

feature-inventory zen=".." cost="benchmarks/per_feature_cost_grid_2026-08-28.tsv":
    mkdir -p target/inventory
    cargo build --release -q -p zenpredict-bake --bin zenpredict-inspect
    cargo run --release -q --example list_features --features hdr -- --variants > target/inventory/universe.txt
    python3 tools/feature_inventory.py \
      --universe target/inventory/universe.txt \
      --cost {{cost}} --cost-class photo --cost-side 2048 \
      --inspect-bin target/release/zenpredict-inspect \
      --label zenjpeg-a-v3-shipped={{zen}}/zenjpeg/zenjpeg/src/encode/picker_data/feature_order.txt \
      --label zenavif-rav1e-v0.1.1-shipped={{zen}}/zenavif/src/models/rav1e_picker_v0_1_1.bin \
      --label zenjpeg-v0.5-modesfull={{zen}}/zenjpeg/benchmarks/zenjpeg_picker_v0.5_modesfull-tiled-evenodd_2026-06-28.manifest.json \
      --label zenjpeg-lossy-ssim2-v0.1={{zen}}/zenjpeg/benchmarks/zenjpeg_lossy_ssim2_picker_v0.1_K3_cleansplit_2026-06-29.manifest.json \
      --label zenavif-lossy-zensim-v0.1={{zen}}/zenavif/benchmarks/pickers/zenavif_lossy_mlp_zensim_v0.1_2026-06-28.manifest.json \
      --label zenjxl-lossy-ssim2-v0.1={{zen}}/zenjxl/benchmarks/zenjxl_lossy_ssim2_picker_v0.1_cleansplit_2026-06-29.manifest.json \
      --label zenjpeg-v2.1-full=zentrain/testdata/zenjpeg_picker_v2.1_full.manifest.json \
      --label zenjxl-v0.7b=benchmarks/zenjxl_picker_v0.7b_2026-05-06.manifest.json \
      --label meta-v0.5-5codec=benchmarks/zenpicker_meta_v0.5_5codec_2026-05-06.manifest.json \
      --out docs/feature-consumption.md

# The torch-dependent zentrain tests (train_hybrid's PyTorch leakyrelu
# student). Kept out of the CI job on purpose — torch is a heavy install —
# so this recipe is the only place they run; a missing torch fails loudly.
zentrain-pytests-torch:
    cd zentrain/tools && uv run --no-project --with pytest --with numpy --with scikit-learn --with torch \
      python -m pytest -q test_train_hybrid_torch_backend.py
