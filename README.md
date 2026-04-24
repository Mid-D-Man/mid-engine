# Mid Engine

> The Middle Man — Modular Anti-Engine

Unity is a black box. Unreal needs unlimited hardware.
Mid is for the Mad Scientists who want a modular,
high-speed toolkit they can actually control.

## Crates

| Crate | Role |
|---|---|
| `mid-log` | Non-blocking tiered logger |
| `mid-net` | Reliable UDP + DixScript networking |
| `mid-ecs` | Data-oriented Entity Component System |
| `mid-math` | SIMD-optimized numerics |
| `mid-common` | Shared types and traits |

## Performance Targets

| System | Frequency |
|---|---|
| Network tick | 128 Hz |
| Physics | 60 Hz |
| Max entities | 100 000+ per core |

## Getting Started

```bash
cargo build
cargo test
cargo run --example headless-server
```

See `docs/` for architecture details and `packets/` for
DixScript packet definitions.
