# Dev Setup

## Local — 2010 MacBook Pro

Develop headless. No GPU required. The old Mac's GPU drivers
are a dead end; the CPU is perfectly capable of running
headless logic, tests, and ECS simulations.

```bash
# Build one crate at a time — saves your CPU
cargo build -p mid-log
cargo build -p mid-net
cargo build -p mid-ecs

# Tests
cargo test -p mid-log
cargo test -p mid-ecs

# Run the headless integration smoke test
cargo run --example headless-server
```

## Cloud Builds — GitHub Actions

Push to main triggers cross-compilation to Linux x86_64.
All heavy Release builds run in Actions — not on the Mac.
This is the correct strategy for limited local hardware.

## Workspace Structure

```
mid-engine/
  crates/
    mid-log/        # Build first — it is the eyes of the engine
    mid-net/        # Depends on mid-common
    mid-ecs/        # Depends on mid-common
    mid-math/       # No internal deps
    mid-common/     # No internal deps — build this first
  tools/
    mdix-compiler/  # Validates packet .mdix files at build time
  examples/
    headless-server/
  packets/          # .mdix packet shape definitions
  docs/
```
