//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
//!
//! The explicit list is every published library crate. `zenpredict-viz` is
//! `publish = false`; `zenpicker-train` is excluded from the workspace and
//! must not be snapshotted; `zentrain` is Python.
//!
//! **`zenanalyze-api` is listed first because it is the crate that matters
//! most here.** It is the version-unifying contract: it freezes at `1.0` and
//! may never break afterwards, since a break splits every consumer that
//! depends on a single `Offer` type unifying across the build. It had been
//! missing from this list — the one published crate in the repo with no
//! committed surface record, which is exactly backwards. Added 2026-08-30
//! while trimming 0.1.1.
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .crates([
            "zenanalyze-api",
            "zenanalyze",
            "zenpredict",
            "zenpicker",
            "zenpredict-bake",
        ])
        .run();
}
