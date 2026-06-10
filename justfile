default:
    cargo test

check:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test

fmt:
    cargo fmt
    cargo test --test public_api_doc

# Regenerate the public-API surface snapshot only (docs/public-api/)
api-doc:
    cargo test --test public_api_doc

# Verify the committed snapshot is current (what CI runs)
api-doc-check:
    ZEN_API_DOC=check cargo test --test public_api_doc

test:
    cargo test

test-verbose:
    cargo test -- --nocapture

clippy:
    cargo clippy --all-targets -- -D warnings
