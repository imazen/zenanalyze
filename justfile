# zenanalyze dev commands

# Format + regenerate the public-API surface snapshots (docs/public-api/)
fmt:
    cargo fmt --all
    cargo test -p zenanalyze --test public_api_doc

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test -p zenanalyze --test public_api_doc

# Verify the committed snapshots are current (what CI runs)
api-doc-check:
    ZEN_API_DOC=check cargo test -p zenanalyze --test public_api_doc
