default:
    cargo test

check:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test

# Format (also regenerates the public-API surface snapshots).
# The snapshot runner lives in the standalone apidoc/ package, so it is
# never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots (docs/public-api/) only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

test:
    cargo test

test-verbose:
    cargo test -- --nocapture

clippy:
    cargo clippy --all-targets -- -D warnings
