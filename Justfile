# Justfile for data-diff-viz

# Build and run with two parquet files
run LEFT RIGHT *ARGS:
    cargo run --release --bin data-diff-viz -- {{LEFT}} {{RIGHT}} {{ARGS}}

# Run all tests (unit + integration)
test:
    cargo test --all-targets

# Lint with clippy (warnings as errors)
lint:
    cargo clippy --all-targets -- -D warnings

# Format
fmt:
    cargo fmt --all

# Format check (CI-friendly)
fmt-check:
    cargo fmt --all -- --check

# Run benchmarks
bench:
    cargo bench

# Generate a directory of demo dataset pairs and launch the visualization
# with the dataset menu populated. Default selection is the fully-matching
# dataset; use the dropdown to switch.
demo:
    rm -rf demo-data
    mkdir -p demo-data
    cargo run --release --bin gen-demo-data -- demo-data
    cargo run --release --bin data-diff-viz -- --demo-dir demo-data
