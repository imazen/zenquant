default:
    cargo test

check:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test

fmt:
    cargo fmt

test:
    cargo test

test-verbose:
    cargo test -- --nocapture

clippy:
    cargo clippy --all-targets -- -D warnings
