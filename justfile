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
      test_train_hybrid_backend.py \
      --deselect test_predict_lib.py::test_student_permutation_relu \
      --deselect test_predict_lib.py::test_student_permutation_leakyrelu

# The torch-dependent zentrain tests (train_hybrid's PyTorch leakyrelu
# student). Kept out of the CI job on purpose — torch is a heavy install —
# so this recipe is the only place they run; a missing torch fails loudly.
zentrain-pytests-torch:
    cd zentrain/tools && uv run --no-project --with pytest --with numpy --with scikit-learn --with torch \
      python -m pytest -q test_train_hybrid_torch_backend.py
