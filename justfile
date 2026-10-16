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

# Validate batching and cancellation without changing test expectations.
check-cancellation:
    cargo test --release --lib --features joint
    cargo test --release --test cancellation --features joint
    cargo clippy --lib --features joint -- -D warnings

# Save stdout from each revision and compare the deterministic output hashes.
fingerprint:
    cargo run --release --example output_fingerprint

# Generated-only output oracle: published 0.1.3 and unchanged ordinary API.
dispatch-parity:
    cargo run --release --manifest-path tests/dispatch-parity/Cargo.toml --bin dispatch-parity
    cargo run --release --manifest-path tests/dispatch-parity/Cargo.toml --bin dispatch-parity --features joint

dispatch-cost:
    cargo run --release --manifest-path tests/dispatch-parity/Cargo.toml --bin unused_cost

# All observations are generated; no corpus or label discovery.
dispatch-check:
    cargo fmt --check
    cargo test --lib
    cargo test --lib --features joint
    cargo clippy --lib --features joint -- -D warnings
    cargo clippy --lib --no-default-features -- -D warnings
    cargo clippy --manifest-path tests/dispatch-parity/Cargo.toml -- -D warnings
    just dispatch-parity

dispatch-api-check:
    cargo semver-checks --package zenquant --baseline-version 0.1.3 --default-features
    cargo semver-checks --package zenquant --baseline-version 0.1.3 --default-features --features joint
    just api-doc
    just api-doc-check

dispatch-cost-analyze log:
    python3 tests/dispatch-parity/analyze_cost.py '{{log}}'
