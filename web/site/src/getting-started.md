# Getting Started

Mid Engine is a Cargo workspace. Most crates build on a plain, current
`stable` toolchain, but a handful need a newer Rust than the workspace's
usual floor — see [Crate Dependency Order](dependency-order.md) and each
crate's own page for specifics before assuming a bare, unqualified command
below will cover everything.

## Build and test

```bash
# Everything a bare-minimum, no-flags toolchain can build:
cargo build
cargo test

# A release build:
cargo test --release

# One crate specifically -- always safe, regardless of any MSRV wall
# elsewhere in the workspace:
cargo build -p mid-math
cargo test  -p mid-math

# The headless server example:
cargo run --example headless-server
```

## A note on `-p`

Several crates deliberately need a newer toolchain than the rest of the
workspace (an upstream dependency's own `edition2024` requirement, mostly —
see each crate's page for the specific wall). A bare `cargo build`/`cargo
test` with no `-p` flag pulls every workspace member in, including those.
Building a single crate with `-p <crate-name>` only resolves that crate's
own dependency closure, so it stays on whatever toolchain that crate
actually needs — usually the workspace floor.

## CI

Every crate's tests run through its own `workflow_dispatch`-only GitHub
Actions workflow — nothing runs automatically on push. Trigger one from the
repository's Actions tab, then check the [Tests](/tests/) page (or the run's
own Job Summary) for results. See [Contributing & CI](contributing.md) for
the full convention.
