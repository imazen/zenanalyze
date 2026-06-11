//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
//!
//! The explicit list mirrors the deliberate selection of the pre-runner
//! snapshot test — the four published library crates. `zenpredict-viz` is
//! `publish = false`; `zenpicker-train` is excluded from the workspace and
//! must not be snapshotted; `zentrain` is Python.
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .crates(["zenanalyze", "zenpredict", "zenpicker", "zenpredict-bake"])
        .run();
}
